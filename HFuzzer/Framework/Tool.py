import json
import os
import time
import httpx
from openai import OpenAI
from datetime import datetime
import random
# from Mutations import Mutations
import requests
import torch

class Tool:
    @staticmethod
    def read_json_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    @staticmethod
    def load_existing_data(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return []
        else:
            return []

    @staticmethod
    def add_dict_json(dict_keyphase, answer_text, power=0.5):
        dict_word = dict_keyphase["phase"]
        dict_power = dict_keyphase["power"]
        old_lists_object = dict_word["object"]
        old_lists_method = dict_word["method"]
        old_lists_add = dict_word["additional"]
        old_lists_power_object = dict_power["power_object"]
        old_lists_power_method = dict_power["power_method"]
        old_lists_power_add = dict_power["power_additional"]

        answer_lists = answer_text.split(";")
        if answer_lists[0].strip() != 'None':
            if answer_lists[0].strip() not in old_lists_object:
                old_lists_object.append(answer_lists[0].strip())
                old_lists_power_object.append(power)
        if len(answer_lists) >1:
            if answer_lists[1].strip() != 'None':
                if answer_lists[1].strip() not in old_lists_method:
                    old_lists_method.append(answer_lists[1].strip())
                    old_lists_power_method.append(power)
        if len(answer_lists) >2:            
            if answer_lists[2].strip() != 'None':
                print(answer_lists[2])
                if answer_lists[2].strip() not in old_lists_add:
                    old_lists_add.append(answer_lists[2].strip())
                    old_lists_power_add.append(power)
        new_dict_word = {
            "object": old_lists_object,
            "method": old_lists_method,
            "additional": old_lists_add
        }
        new_dict_power = {
            "power_object": old_lists_power_object,
            "power_method": old_lists_power_method,
            "power_additional": old_lists_power_add
        }
        new_dict_data = {
            "phase": new_dict_word,
            "power": new_dict_power
        }
        return new_dict_data

    @staticmethod
    def save_answer_json(output_data, file_path):
        existing_data = Tool.load_existing_data(file_path)
        existing_data.append(output_data)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)
    
    # @staticmethod
    # def check_package_exists(package_name):
    #     url = f"https://pypi.org/pypi/{package_name}/json"
    #     while(1):  # Try up to 3 times
    #         try:
    #             response = requests.get(url)
    #             return response.status_code == 200
    #         except Exception as e:
    #             print(f"Request failed for {package_name}: {e}")
    #     return False
    
    @staticmethod
    def check_package_exists(valid_packages, package_name):
        return package_name in valid_packages

    @staticmethod
    def get_first_release_date(package_name):
        url = f"https://pypi.org/pypi/{package_name}/json"
        while(1):  # Try up to 3 times
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    releases = data.get("releases", {})
                    sorted_releases = sorted(
                        (release, details[0]["upload_time"])
                        for release, details in releases.items()
                        if details
                    )
                    if sorted_releases:
                        first_release, upload_time = sorted_releases[0]
                        return first_release, upload_time
                    else:
                        return None, "No releases found"
                else:
                    return None, "Failed to fetch package info for " + package_name
            except requests.RequestException as e:
                print(f"Request failed for {package_name}: {e}")
        return False, "Request failed"

    @staticmethod
    def save_no_releases_to_json(data, output_file):
        try:
            with open(output_file, mode='w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

        except Exception as e:
            raise

    @staticmethod
    def check_package_source(package_name):
        base_url = f'https://libraries.io/api/search'
        params = {
            'q': package_name,
            'api_key': ""  # please provide your own API key
        }    
        platform_list = set()
        while(1):  # Try up to 3 times
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if not item.get('name') == package_name:
                            continue
                        platform_list.add(item.get('platform'))
                    break  # Exit the loop if the request was successful
                else:
                    return {'error': f'Failed to fetch data: {response.status_code}'}
            except requests.exceptions.SSLError:
                continue  # Retry on SSL error
        else:
            return {'error': 'Failed to fetch data after 3 attempts due to SSL errors'}
        return platform_list

    @staticmethod
    def get_response_api(messages, temperature):
        client = OpenAI(
            api_key="", # fill in your own api key
            base_url="https://api.deepseek.com",  # DeepSeek的API端点
        )

        while(1):  # Try up to 3 times
            try:
                completion = client.chat.completions.create(
                    model="deepseek-chat",  
                    messages=messages,
                    temperature=temperature
                )
                return completion
            except Exception as e:
                print(f"HTTPX request error: {e}")
                time.sleep(2)
        return None

    @staticmethod
    def get_description(package_name):
        BASE_URL = f"https://libraries.io/api/pypi/{package_name}?api_key=libraries_key" # please provide your own API key
        description = ""
        while(1):
            try:
                response = requests.get(BASE_URL)
                if response.status_code == 404:
                    return "package_not_found"
                else:
                    response.raise_for_status()
                    data = response.json()
                    description = data["description"]
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for package {package_name}: {e}")
                time.sleep(2)
        return description

    @staticmethod
    def get_response_local(model_name, messages, temperature,openai_api_base):
        openai_api_key = "EMPTY"
        client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
        )
        while(1):  # Try up to 3 times
            try:
                # print(messages)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=3000
                )
                return completion
            except Exception as e:
                print(f"HTTPX request error: {e}")
                time.sleep(2)
        return None
      
    @staticmethod
    def get_response_transformer(model, tokenizer, messages, temperature):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length - 3000  
        ).to(model.device) 
        
        if temperature > 1e-5:
            generate_kwargs = {
                "temperature": temperature,
                "max_new_tokens": 3000,
                "do_sample": True,  
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "top_p": 0.9 if temperature > 1e-5 else 1.0,  
            }
        else:
            generate_kwargs = {
                "max_new_tokens": 3000,
                "do_sample": False,  
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "top_p": 0.9 if temperature > 1e-5 else 1.0,  
            }
       
        with torch.no_grad():  
            start_time = time.time()
            outputs = model.generate(**inputs,** generate_kwargs)
            end_time = time.time()
        
        generate_time = end_time - start_time
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockChoice:
            def __init__(self, message):
                self.message = message
        
        class MockCompletion:
            def __init__(self, choices):
                self.choices = choices
        
        return MockCompletion([MockChoice(MockMessage(response_text))])
    
    @staticmethod
    def get_response_packmonitor(model, messages, temperature, use_packmonitor=False, package_path=None):
        start_generate = time.time()         
        response_text, total_decoded_tokens = model.generate(
            messages=messages,
            # package_path=package_path,
            use_packmonitor=use_packmonitor,
            max_tokens=3000,
            temperature=temperature if temperature > 1e-5 else 1e-5
        )
        end_generate = time.time()
        generation_time = end_generate - start_generate
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockChoice:
            def __init__(self, message):
                self.message = message
        
        class MockCompletion:
            def __init__(self, choices):
                self.choices = choices
        
        return MockCompletion([MockChoice(MockMessage(response_text))]), generation_time, total_decoded_tokens