import json
import time
import numpy as np
import torch
import random
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from llguidance import grammar_from, hf, LLMatcher, LLParserLimits
import argparse

class PackMonitor:
    def __init__(self, model_name: str, package_path: Optional[List[str]] = None, device: str = "cuda"):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True,
        )
        
        self.ll_tokenizer = hf.from_tokenizer(
            self.hf_tokenizer, n_vocab=self.model.config.vocab_size
        )
        self.device = self.model.device
        self.model.eval()
        self.matcher_cache: dict = {}
        if package_path :
            self.package_list = self.load_packages(package_path)
        else :
            self.package_list = None
        
    def get_cached_matcher(self, package_list):
        cache_key = tuple(sorted(package_list))

        if cache_key in self.matcher_cache:
            cached = self.matcher_cache[cache_key]
            base_matcher = cached["matcher"]
            limits = cached["limits"]
            return base_matcher.deep_copy(), limits

        grammar = self.make_grammar(package_list)
        g = grammar_from("lark", grammar)

        limits = None
        if len(package_list) > 1000:
            estimated_fuel = len(package_list) * 100
            limits = LLParserLimits(
                initial_lexer_fuel=max(estimated_fuel, 500_000_000),
                step_lexer_fuel=5_000_000,
                max_lexer_states=2_000_000,
                max_grammar_size=10_000_000,
            )

        matcher = LLMatcher(self.ll_tokenizer, grammar=g, limits=limits)
        if matcher.is_error():
            print(f"Grammar error: {matcher.get_error()}")
            return None, limits

        self.matcher_cache[cache_key] = {"matcher": matcher, "limits": limits}
        return matcher.deep_copy(), limits

    def load_packages(self, file_paths: List[str]) -> List[str]:
        all_packages = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_packages.extend(data)
                
            except Exception as e:
                print(f"Warning: 加载文件 {file_path} 时出错: {e}")
        
        unique_packages = sorted(list(set(all_packages)))
        print(f"Total valid packages:{len(unique_packages)}")
        return unique_packages
    
    def sample(self, masked_logits, temperature):
        if temperature > 0:
            probs = torch.softmax(masked_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(masked_logits, dim=-1, keepdim=True)
            
        return next_id

    def make_grammar(self, package_list: List[str]) -> str:
        
        if not package_list:
            return r"""%llguidance { "allow_invalid_utf8": true }
                    start: /(\n|.)*/"""

        pkgs = [p.replace('"', r'\"') for p in package_list]
        valid_packages = " | ".join(f'"{p}"' for p in pkgs)

        grammar = rf"""
            %llguidance {{ "allow_invalid_utf8": true }}
            PACKAGE_NAME: {valid_packages}
            WS: /[ \t]+/
            PACKAGE_NAMES: PACKAGE_NAME (WS PACKAGE_NAME)*
            PIP_INSTALL_PREFIX: /\s*pip\s+install\s+/
            INSTALL_CMD_LINE: (PIP_INSTALL_PREFIX) PACKAGE_NAMES /\n/
            ORIDINARY_CODE_LINE: /[^\n]*/ /\n/ & ~/(?s:.*)(pip\s+install)(?s:.*)/
            CODE_BODY: (INSTALL_CMD_LINE | ORIDINARY_CODE_LINE)*
            CODE_START: "```bash\n"
            CODE_END: "\n```"
            CODE_BLOCK: CODE_START CODE_BODY CODE_END
            NATURAL_LANG: /[^`]*/
            start: (NATATURAL_LANG | CODE_BLOCK)*
            """
        
        return grammar.strip()

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict],
        use_packmonitor: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        full_prompt = self.hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_output = self.hf_tokenizer(full_prompt, return_tensors="pt")
        input_ids = encoded_output['input_ids'].to(self.device)
        input_len = input_ids.shape[1]
        prompt_ids = input_ids.squeeze(0).tolist()
        prompt_len = input_ids.shape[-1]

        if not use_packmonitor:
            return self.vanilla_generate(
                input_ids, max_tokens, temperature
            )

        package_list = self.package_list
        
        matcher = None
        start_building = time.time()
        matcher, limits = self.get_cached_matcher(package_list)
        end_building = time.time()
        print(f"building_time:{end_building - start_building}s")
        
        if matcher is None:
            return self.vanilla_generate(input_ids, max_tokens, temperature)

        if matcher.is_error():
            error_msg = matcher.get_error()
            print(f"Grammar error: {error_msg}")
            return self.vanilla_generate(
                input_ids, max_tokens, temperature
            )
        
        eos_token_id = self.hf_tokenizer.eos_token_id
        start_generate = time.time()
        for step in range(max_tokens):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  
            # 获取 logit bias（0=禁用，200=允许）
            bias_bytes = matcher.compute_logit_bias()
            
            bias_np = np.frombuffer(
                bias_bytes, dtype=np.uint8, count=self.model.config.vocab_size
            )
            
            bias = torch.tensor(bias_np, device=self.device, dtype=logits.dtype)
            # 0 -> -inf，200 -> 0
            bias = torch.where(
                bias > 0, torch.zeros_like(bias), torch.full_like(bias, float("-inf"))
            )
            masked_logits = logits + bias

            next_id = self.sample(masked_logits, temperature)
            next_id_int = next_id.item()

            if eos_token_id is not None and next_id_int == eos_token_id:
                input_ids = torch.cat([input_ids, next_id], dim=1)
                break
    
            if not matcher.consume_token(next_id_int):
                if matcher.is_error():
                    print(f"Error consuming token {next_id_int}: {matcher.get_error()}")
                break

            input_ids = torch.cat([input_ids, next_id], dim=1)

            if matcher.is_stopped():
                break
        end_generate = time.time()
        print(f"generate time:{end_generate - start_generate}s")
        total_decoded_tokens = input_ids.shape[1] - input_len
        return self.hf_tokenizer.decode(input_ids[0][input_len:], skip_special_tokens=True), total_decoded_tokens

    @torch.inference_mode()
    def vanilla_generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float
    ) -> str:
        eos_token_id = self.hf_tokenizer.eos_token_id
        input_len = input_ids.shape[1]
        
        for step in range(max_tokens):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  

            next_id = self.sample(logits, temperature)
            next_id_int = next_id.item()

            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_token_id is not None and next_id_int == eos_token_id:
                break

        total_decoded_tokens = input_ids.shape[1] - input_len
        return self.hf_tokenizer.decode(input_ids[0][input_len:], skip_special_tokens=True), total_decoded_tokens
    

