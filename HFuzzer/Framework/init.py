
import json
from Tool import Tool
import os
import re 
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def run(model_name,temperature,target_model_name,target_temperature,timestr, use_packmonitor=False, use_RAG=False, use_SR=False):
    if use_packmonitor:
        BASE_PATH = "../results_packmonitor"
    else:
        BASE_PATH = "../results_vanilla"
    SAVE_PATH = BASE_PATH 
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)


    fold_model_name = model_name.split("/")[-1]
    fold_target_model_name = target_model_name.split("/")[-1]

    if use_packmonitor:
        fold_name = "PackMonitor_"+fold_model_name+"_"+str(temperature)+"__"+fold_target_model_name+"_"+str(target_temperature)
    else:
        fold_name = "Vanilla_"+fold_model_name+"_"+str(temperature)+"__"+fold_target_model_name+"_"+str(target_temperature)

    if use_RAG:
        fold_name = fold_name + "_RAG"
        
    if use_SR:
        fold_name = fold_name + "_SR"
        
    SAVE_PATH = os.path.join(SAVE_PATH, fold_name)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    pattern = r"\[ANSWER\](.*?)\[/ANSWER\]"

    desc_path = os.path.join("../Framework", "desc_pypi.json")
    with open(desc_path, 'r', encoding='utf-8') as file:
        try:
            existing_data = json.load(file)
        except json.JSONDecodeError:
            existing_data = []

    dict_keyphase = {
        "phase": {
            "object": [],
            "method": [],
            "additional": []
        },
        "power":{
            "power_object": [],
            "power_method": [],
            "power_additional": []
        }
    }
    if temperature.is_integer():  
        temperature_str = str(int(temperature))
    else:
        temperature_str = str(temperature)
        
    extracter_path = os.path.join(SAVE_PATH, f"extracter_{fold_model_name}_{temperature}.json") 
    dict_path = os.path.join(SAVE_PATH, f"dict_{fold_model_name}_{temperature}.json") 
    if os.path.exists(dict_path):
        return 
    for item in existing_data:
        name = item['name']
        desc = item['description']
        messages = [
            {
                "role": "system",
                "content":
                    '''You are a key phrase extractor who can extract phrases from the package description to describe the package's functionality.
                    You only need to reply with three key phrases, separated by ';', without adding any other text or formatting.
                    The first phrase is used to indicate the object that the package handles, the second phrase is used to indicatethe method that the package handles the object, and the third phrase is used to append additional information.
                    If you think the package description does not have relevant information, replace the phrase with 'None'.
                    Provide your answer in [ANSWER] and [/ANSWER] tags, following the examples.
                    Examples:
                    [NAME]
                    hashio
                    [/NAME]
                    [DESCRIPTION]
                    Custom file and directory checksum tool
                    [/DESCRIPTION]
                    [ANSWER]
                    file and directory; checksum; custom tool
                    [/ANSWER]'''
                ,
            },
            {
                "role": "user",
                "content": f'''[NAME]
                {name}
                [/NAME]
                [DESCRIPTION]
                {desc}
                [/DESCRIPTION]''',
                        },
        ]

        completion = Tool.get_response_api(messages, temperature)
        answer_text_p = completion.choices[0].message.content
        answer_text = re.findall(pattern, answer_text_p, re.DOTALL)
        if len(answer_text) ==1:
            answer_text = answer_text[0]
        else:
            answer_text = answer_text_p
        answer_dict = {
            "question":messages,
            "completion":answer_text
        }
        Tool.save_answer_json(answer_dict,extracter_path) 
        dict_keyphase = Tool.add_dict_json(dict_keyphase, answer_text, 1.0) 
        with open(dict_path, 'w', encoding='utf-8') as file:
            json.dump(dict_keyphase, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified parameters.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature setting for the model")
    parser.add_argument("--target_model_name", type=str, required=True, help="Target model name")
    parser.add_argument("--target_temperature", type=float, required=True, help="Target model temperature")
    parser.add_argument("--timestr", type=str, required=True, help="Timestamp")
    parser.add_argument("--use_packmonitor", action='store_true', help="Whether use packmonitor")
    parser.add_argument("--use_RAG", action='store_true', help="Whether use RAG")
    parser.add_argument("--use_SR", action='store_true', help="Whether use self-refinement")

    args = parser.parse_args()
    run(args.model_name, args.temperature, args.target_model_name, args.target_temperature, args.timestr, args.use_packmonitor, args.use_RAG, args.use_SR)