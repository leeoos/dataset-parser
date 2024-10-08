import os
import re
import argparse
import logging
from datasets import (
    load_dataset,
    load_dataset_builder,
    get_dataset_config_names
)
#import pandas as pd
#import processor  

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/main.log', encoding='utf-8', level=logging.DEBUG, filemode='w')

# Path to the dataset list
DATASETS_LIST_PATH = "filtered_links.csv"
OUTPUT_DIR = "output"
LOCAL_CACHE = "local_cache"


def extract_configs(message):

    # Use regular expression to find the part inside square brackets
    config_list = re.findall(r"'([^']*it[^']*)'", message)
    if config_list:
        return config_list  # Return the list with 'it' matches
    else:
        # If no matches containing 'it', return all configs
        config_list = re.findall(r"\['(.*?)'\]", message)
        return config_list


def download_hf_datasets(hf_link, save_path='./'):
    datasets = []

    # Get the configurations for the dataset
    configs = get_dataset_config_names(hf_link)
    logger.info(f"Available configurations: {configs}")

    # Download dataset
    try:
        print("Downloading datasets...")
        for config in configs:
            datasets.append(load_dataset(hf_link,  config, cache_dir=save_path))
        return datasets
    
    except Exception as e:
        logger.debug(f"Failed to load dataset {hf_link}: {e}")
        return None


def process_hf_datasets(name=None, task=None):

    # Read the dataset list
    with open(DATASETS_LIST_PATH, "r") as file:
        datasets_list = [line.strip() for line in file if line.strip()]

    # Iterate over datasets
    first_seen = True
    for idx, dataset in enumerate(datasets_list):
        dataset = dataset.split(',')
        task_type = dataset[0]
        
        if task_type != task:
            if not first_seen: break
            pass

        else:
            first_seen = False
            dataset_name = dataset[1].split("datasets/")[-1]

            if name and dataset_name.split('/')[1] == name:
                logger.info(f"Selected dataset --> {dataset_name}\tindex --> {idx}")
                acquired_datasets = download_hf_datasets(dataset_name, save_path=LOCAL_CACHE)
                if acquired_datasets: logger.info(f"{dataset_name} --> Downloaded")

                

                break

            if not name: 
                # Do not break
                save_path = LOCAL_CACHE + "/" + args.name
                if os.path.exists(save_path) and os.listdir(save_path) is not []:
                    ...
                else:
                    logger.info(f"Selected dataset --> {dataset_name}\tindex --> {idx}")
                    acquired_datasets = download_hf_datasets(dataset_name, save_path=LOCAL_CACHE)

            # logger.info(f"{idx} - Processing dataset: {dataset_name}")

#         # Apply custom processing (defined in processor.py)
#         processed_data = processor.process_dataset(dataset)
#
#         # Save the processed dataset to CSV or other formats
#         save_processed_dataset(dataset_name, processed_data)


def save_processed_dataset(dataset_name, processed_data):
    # Define the path to save the processed dataset
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name.replace('/', '_')}_processed.csv")

    # Assume processed_data is a pandas DataFrame or similar
    processed_data.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")


if __name__ == "__main__":

    # Parse script args
    parser = argparse.ArgumentParser(
        prog='Main',
        description='Select different types of datasets to parse for instruction tuning of minerva'
    )

    parser.add_argument('-n', '--name', type=str, help='insert the name of the datasetto parse')
    parser.add_argument('-e', '--extension', type=str, help='insert the format extension of the dataset to parse')
    parser.add_argument('-t', '--task', type=str, help='insert the name of the task')
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

    if args.huggingface:
        process_hf_datasets(name=args.name, task=args.task)

    elif args.local:
        ...

