import os
import re
import csv
import json
import tqdm
import time
import logging
import argparse
import numpy as np
from datasets import (
    load_dataset, 
    load_from_disk,
    load_dataset_builder, 
    get_dataset_config_names
)

# import pandas as pd
from hf_processor import format

# Logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/main.log", 
    encoding="utf-8", 
    level=logging.DEBUG, 
    filemode="w"
)
logger = logging.getLogger(__name__)
    
# Path to the dataset list
DATASETS_LIST_PATH = "huggingface_links.csv"
OUTPUT_DIR = "output"
LOCAL_CACHE = "local_cache"
LOCAL_DATA = "local_data"
TASK_INFO = "dataset_info"
DATASET_DICTIONAY = {}

# Download data from huggingface repo
def download_hf_datasets(name, hf_link, save_path="./", cache_path="./"):

    # Get the configurations for the dataset
    datasets = {}
    try:
        configs = get_dataset_config_names(
            hf_link,
        )
        filgtered_configs = list(filter(lambda x: "it" in x.lower(), configs))
        configs = filgtered_configs if filgtered_configs != [] else configs
        logger.info(f"Dataset: {hf_link}\tAvailable configurations: {configs}")

        # Save configurations
        configs = np.array(configs)
        np.savez(save_path + "/configs", configs)
        np.savez(cache_path + "/configs", configs)
            
        # Download dataset
        try:
            print()
            datasets = {
                config: load_dataset(
                    hf_link, 
                    config, 
                    cache_dir=cache_path
                ) 
                for config in configs
            }

            # Save Data
            logger.info(f"Saving data in : {save_path}")
            for key, value in datasets.items():
                value.save_to_disk(save_path + "/" + key)

            return datasets

        except Exception as e:
            logger.debug(f"Failed to load dataset {hf_link}: {e}")
            return datasets

    except Exception as e:
        logger.debug(f"Failed to load informations for  {hf_link}: {e}")
        return datasets


# Load data from local directory
def load_hf_dataset_from_local(name, save_path):
    
    configs = np.load(save_path + "/configs.npz")['arr_0']
    logger.info(f"Loading data from: {save_path} using: {configs}")
    
    # Load dataset
    datasets = {}
    try:
        datasets = {
            config: load_from_disk(save_path + "/" + config)            
            for config in configs
        }
        return datasets

    except Exception as e:
        logger.debug(f"Failed to load dataset {name}: {e}")
        return datasets  


# Load data from cache directory
def load_hf_dataset_from_cache(hf_link, cache_path):
    
    configs = np.load(cache_path + "/configs.npz")['arr_0']
    logger.info(f"Loading data from: {cache_path} using: {configs}")
    
    # Load dataset
    datasets = {}
    try:
        datasets = {
            config: load_dataset(
                hf_link,
                config,
                cache_dir=cache_path,
                download_mode="reuse_dataset_if_exists"
            )            
            for config in configs
        }
        return datasets

    except Exception as e:
        logger.debug(f"Failed to load dataset {hf_link}: {e}")
        return datasets  

# Collect datasets into a global dictionary
def add_dataset_to_dictionary(dataset, dataset_name):
    
    logger.info(f"dataset loaded: {dataset != {}}")
    if dataset:
        DATASET_DICTIONAY[dataset_name] = dataset
        print("ok")
    else:
        print("fail")


# Acquire and process Hugging face datasets
def process_hf_datasets(name=None, task=None):

    # Read the dataset list
    print("Retriving datasets...")
    start_time = time.time()
    
    with open(DATASETS_LIST_PATH, "r") as file:
        datasets_list = csv.reader(file)
        next(datasets_list, None)  # skip header

        # Iterate over datasets
        first_seen_task = True
        for idx, selected_dataset in enumerate(datasets_list):
            
            # Skip all unwated datasets
            task_type = selected_dataset[0]
            if task_type != task and not first_seen_task: break
            if task_type != task: continue
            
            # Get dataset informations
            dataset_name = selected_dataset[1]
            dataset_link = selected_dataset[2].split("datasets/")[-1]
            cache_path = LOCAL_CACHE + "/" + dataset_name
            save_path = LOCAL_DATA + "/" + dataset_name

            # Select specific dataset
            if name and dataset_name != name: continue
            first_seen_task = False

            os.makedirs(save_path, exist_ok=True)
            os.makedirs(cache_path, exist_ok=True)
            print(f"{dataset_name}...", end='\t\t')

            if os.listdir(save_path) != []:
                # Load dataset from save_path
                dataset = load_hf_dataset_from_local(dataset_name, save_path)
                print("using local..." , end='\t\t')

            elif os.listdir(cache_path) != []:
                # Load dataset from cache_path
                dataset = load_hf_dataset_from_cache(dataset_link, cache_path)
                print("using cache..." , end='\t\t')

            else:
                # Download dataset from repo
                dataset = download_hf_datasets(
                    dataset_name,
                    dataset_link, 
                    save_path,
                    cache_path
                )
                logger.info(f"{dataset_name} status: Downloaded")

            add_dataset_to_dictionary(dataset, dataset_name)
            
            if name and dataset_name == name: break


    print("\n--- %s seconds ---" % (time.time() - start_time))
    logger.info("execution time: %s seconds " % (time.time() - start_time))

    print("Formatting acquired datasets...")
    for name, dataset in DATASET_DICTIONAY.items():
        format(
            dataset, 
            name, 
            task, 
            TASK_INFO, 
            logger
        )


if __name__ == "__main__":
    
    # Parse script argsreader
    parser = argparse.ArgumentParser(
        prog="Main",
        description="Select different types of datasets to parse for instruction tuning of minerva",
    )

    parser.add_argument(
        "-n", "--name", type=str, help="insert the name of the datasetto parse"
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        help="insert the format extension of the dataset to parse",
    )
    parser.add_argument(
        "-t", "--task", type=str, help="insert the name of the task", required=True
    )
    parser.add_argument("-hf", "--huggingface", action="store_true")
    parser.add_argument("-l", "--local", action="store_true")

    args = parser.parse_args()

    # Setup directories
    if args.task:
        OUTPUT_DIR += "/" + args.task
        LOCAL_CACHE += "/" + args.task
        LOCAL_DATA += "/" + args.task

    if args.name:
        OUTPUT_DIR += "/" + args.name

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    os.makedirs(LOCAL_DATA, exist_ok=True)

    # Setup task info
    if args.task.upper() == "NER":
        TASK_INFO = TASK_INFO + "/" + "ner"
        os.makedirs(TASK_INFO, exist_ok=True)

    if args.huggingface:
        dataset = process_hf_datasets(name=args.name, task=args.task)

    elif args.local:
        ...
