import glob
import numpy as np
import os
import random
import shutil
import uuid

def get_template(models:list, dataset:str="21", config:str="3d_fullres", tr:str='ourTrainer')->str:
    partition = random.choice(['gpu-v100', 'gpu-a100'])
    template = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=32GB
#SBATCH --gres=gpu:{(2 if len(models) > 1 else 1)}

#Environment setup
source ~/software/init-conda
conda activate torch_env
#Commands
"""
    for i, model in enumerate(models):
        model = model.replace("andrew.heschl", "$USER")
        template += f"CUDA_VISIBLE_DEVICES={(0 if i%2 == 0 else 1)} nnUNetv2_train {dataset} {config} 0 -tr {tr} -model {model}"
        if i%2 != 0:
            template += "\n\nwait\n\n"
        else:
            template += " &\n"

    return template

def make_folders(quantity:int, local_path:str)->None:
    try:
        shutil.rmtree(f"{local_path}/distribution")
    except:
        pass
    try:
        os.mkdir(f"{local_path}/distribution")
    except:
        pass

    for i in range (quantity):
        try:
            os.mkdir(f"{local_path}/distribution/{i}")
        except:
            pass

task_name = ""
def save_file(content:str, id:int, local_path:str):
    global task_name
    if task_name == "":
        task_name = str(input("Task_name: "))
    file_name = task_name + "_" + str(uuid.uuid1())[0:8]
    with open(f"{local_path}/distribution/{id}/{file_name}.slurm", "w+") as target:
        target.write(content)

def write_jobs(models:list, id:int, local_path:str, configuration:str="3d_fullres")->None:
    target_files = []
    i = 0
    k = 4
    while i < len(models):
        target_files.append(models[i:min(len(models), i+k+1)])
        i = min(len(models), i+k+1)
    for task in target_files:
        content = get_template(task)
        save_file(content, id, local_path)

if __name__ == "__main__":

    model_folder = str(input("Model folder to run: "))
    local_path = '/'.join(__file__.split('/')[0:-1])
    models = glob.glob(f"{model_folder}/**/*.json", recursive=True)

    num_splits = int(input("How many splits? "))
    config = str(input("What configuration? "))
    tasks = np.array_split(models, num_splits)

    make_folders(num_splits, local_path)
    models = np.array_split(models, num_splits)
    for id in range(num_splits):
        write_jobs(models[id], id, local_path, config)
    
    print(f"Success! Tasks distributed to {local_path}/distribution")
    
