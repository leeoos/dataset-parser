import os
import re
import csv
import tqdm
import time
import logging
import argparse
from datasets import (
    load_dataset,
    load_dataset_builder,
    get_dataset_config_names
)
#import pandas as pd
import processor  

# Logger
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/main.log', encoding='utf-8', level=logging.DEBUG, filemode='w')

# Path to the dataset list
DATASETS_LIST_PATH = "huggingface_links.csv"
OUTPUT_DIR = "output"
LOCAL_CACHE = "local_cache"
TASK_INFO = "dataset_info"


def download_hf_datasets(hf_link, save_path='./'):
    datasets = []

    # Get the configurations for the dataset
    try:
        configs = get_dataset_config_names(hf_link, )
        filgtered_configs = list(filter(lambda x: 'it' in x.lower(), configs))
        configs = filgtered_configs if filgtered_configs != [] else configs
        logger.info(f"Dataset: {hf_link}\tAvailable configurations: {configs}")
        
        # Download dataset
        try:
            datasets = [load_dataset(hf_link,  config, cache_dir=save_path) for config in configs]
            return datasets
        
        except Exception as e:
            logger.debug(f"Failed to load dataset {hf_link}: {e}")
            return datasets
        
    except Exception as e:
        logger.debug(f"Failed to load informations for  {hf_link}: {e}")
        return datasets


def process_hf_datasets(name=None, task=None):
    acquired_datasets = []

    # Read the dataset list
    print("Retriving datasets...")
    start_time = time.time() 
    with open(DATASETS_LIST_PATH, "r") as file:
        datasets_list = csv.reader(file)
        next(datasets_list, None) # skip header

        # Iterate over datasets
        first_seen = True
        for idx, selected_dataset in enumerate(datasets_list):
            task_type = selected_dataset[0]
            dataset_name = selected_dataset[1]
            dataset_link = selected_dataset[2].split("datasets/")[-1]
            save_path = LOCAL_CACHE + "/" + dataset_name
            os.makedirs(save_path, exist_ok=True)

            if task_type != task:
                if not first_seen: break
                pass
            else:
                first_seen = False

                if name and dataset_name == name:
                    print(f"{dataset_name}...", end='\t\t')

                    # skip already processed dataset
                    if (os.listdir(save_path) != []): 
                        print("ok")
                        continue

                    logger.info(f"Selected dataset: {dataset_name}\tindex: {idx}")
                    acquired_dataset = download_hf_datasets(dataset_link, save_path=save_path)

                    if acquired_datasets: 
                        logger.info(f"{dataset_name} status: Downloaded")
                        acquired_datasets.append(acquired_dataset)
                        print("ok")
                    else:
                        print("fail")
                    break

                elif not name:   
                    print(f"{dataset_name}...", end='\t\t')

                    # skip already processed dataset
                    if (os.listdir(save_path) != []): 
                        print("using cache... \tok")
                        continue    

                    logger.info(f"Selected dataset: {dataset_name}\tindex: {idx}")
                    acquired_dataset = download_hf_datasets(dataset_link, save_path=save_path)
                    
                    if acquired_datasets: 
                        acquired_datasets.append(acquired_dataset)
                        print("ok")
                    else: 
                        print("fail")

    print("\n--- %s seconds ---" % (time.time() - start_time))
    logger.info("execution time: %s seconds " % (time.time() - start_time))

    for dataset in acquired_datasets:
        processor.format(dataset, TASK_INFO)

    return acquired_datasets


if __name__ == "__main__":

    # Parse script argsreader
    parser = argparse.ArgumentParser(
        prog='Main',
        description='Select different types of datasets to parse for instruction tuning of minerva'
    )

    parser.add_argument('-n', '--name', type=str, help='insert the name of the datasetto parse')
    parser.add_argument('-e', '--extension', type=str, help='insert the format extension of the dataset to parse')
    parser.add_argument('-t', '--task', type=str, help='insert the name of the task', required=True)
    parser.add_argument('-hf', '--huggingface', action='store_true')
    parser.add_argument('-l', '--local', action='store_true')
    
    args = parser.parse_args()

    # Setup directories
    if args.task: 
        OUTPUT_DIR += "/" + args.task
        LOCAL_CACHE += "/" + args.task
    if args.name: OUTPUT_DIR += "/" + args.name

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    os.makedirs(TASK_INFO, exist_ok=True)

    # Setup task info
    if args.task.upper() == "NER":
        info_file = TASK_INFO + "/" + "ner_tags"

    if args.huggingface:
        dataset = process_hf_datasets(name=args.name, task=args.task)

    elif args.local:
        ...

