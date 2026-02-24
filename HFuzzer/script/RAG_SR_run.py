import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_framework(temperature, target_temperature, item_attack, item_asker, timestr):
    LOG_PATH = "../log"
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    log_file_frame = f"{LOG_PATH}/SR_{item_attack.split('/')[-1]}_{item_asker.split('/')[-1]}.log"
    script_path_frame_initial = "../Framework/init.py"
    command_frame_initial = [
        "python", script_path_frame_initial,
        "--model_name", item_attack,
        "--temperature", str(temperature),
        "--target_model_name", item_asker,
        "--target_temperature", str(target_temperature),
        "--timestr", timestr,
        "--use_SR",
        "--use_RAG"
    ]
    try:
        with open(log_file_frame, "w") as log_file:
            result = subprocess.run(command_frame_initial, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)
        print(f"Script executed successfully. Log saved to: {log_file_frame}")
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed. Check the log file: {log_file_frame}")
    script_path_frame_loop = "../Framework/loop.py"
    command_frame_loop = [
        "python", script_path_frame_loop,
        "--model_name", item_attack,
        "--temperature", str(temperature),
        "--target_model_name", item_asker,
        "--target_temperature", str(target_temperature),
        "--timestr", timestr,
        "--use_SR",
        "--use_RAG"
    ]
    try:
        with open(log_file_frame, "a+") as log_file:
            result = subprocess.run(command_frame_loop, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)
        print(f"Script executed successfully. Log saved to: {log_file_frame}")
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed. Check the log file: {log_file_frame}")


def run_parallel(temperature,target_temperature, item_attack, item_asker, timestr):
    with ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(run_framework, temperature, target_temperature, item_attack, item_asker, timestr))
        for future in futures:
            future.result()

list_attack_name = [
    
]

dict_asker_name = {
  "path_to_your_local_model": "model_release_date"
}


temperature = 0  
target_temperature = 0.7
for item_attack in list_attack_name:
    for item_asker,timestr in dict_asker_name.items():
        print("Attack model:", item_attack)     
        print("Asker model:", item_asker)

        run_parallel(temperature, target_temperature, item_attack, item_asker, timestr)