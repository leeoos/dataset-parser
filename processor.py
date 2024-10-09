#!/usr/bin/env python

# general imports 
import os
import sys
import shutil
import argparse
# from git import Repo

# data manipulations
import json
import csv
import pandas as pd
import numpy as np


def format(dataset, task, dataset_info=None): 
    
    # Select the appropriate task 
    if task.upper() == 'NER':
        format_ner()


def format_ner(dataset, dataset_info):
    ...
#
# def move_data(data, dest="results"):
#   """Just to move the final dtaset in the destination folder and once again avoid pushing large files."""
#
#   os.makedirs(dest, exist_ok=True)
#   for d in data:
#     shutil.move(d, dest + "/" + d)
#     print(f"Data: {d} moved to {dest}")
#
#
# def make_dataframe(dataset, dtype):
#   """This finction read the TSV file while preserving empty lines and return a pandas dataframe"""
#
#   header_names = ["word", "label"]
#   with open(dataset, "r", encoding="utf-8") as f:
#     try:
#       df = pd.read_csv(f, sep='\t', header=None, skip_blank_lines=False, names=header_names, quoting=csv.QUOTE_NONE)
#       return df
#     except Exception as e:
#       print(f"An error occurred: {e}")
#
#
# def join_strings_smartly(words):
#     """ Joins a list of words smartly:
#     - Adds spaces between words when appropriate.
#     - Avoids adding spaces before punctuation.
#     """
#     punctuation = {'.', ',', ';', ':', '!', '?'}
#     result = words[0]
#     prev = result
#
#     for word in words[1:]:
#       if word in punctuation or \
#         "'" in prev or \
#         word.startswith("'") or \
#         ("." in prev and "." in word) :
#         # add word without space
#         result += word
#       else:
#         # add with space
#         result += " " + word
#       # keep track of previous word  
#       prev = word
#
#     return result
#
#
# def make_json(output_jsonl, sentence_id, sentence_text, entities, DEBUG=False):
#   """This function is used to generate a desired json object. Each samples as an array of choices associated to it:
#   PER --> person
#   ORG --> organization
#   LOC --> location
#   """
#
#   choices = ["persona", "organizzazione", "luogo"]
#   with open(output_jsonl, "a",  encoding="utf-8") as jout:
#
#     for ent in entities:
#       has_line = bool(jout.tell())
#       entity_name = ent[0]
#       label = ent[1]
#
#       json_dict ={
#         "sentence_id": sentence_id, 
#         "text": sentence_text,
#         "target_entity": entity_name,
#         "choices": choices,
#         "label": label
#       }
#
#       json_str = json.dumps(json_dict, ensure_ascii=False)
#       # if the file already contains some lines the insert newline char before
#       if bool(has_line):
#         jout.write("\n" + json_str)
#       else:
#         jout.write(json_str)
#
#       # jout.write(json_str + '\n')
#
#
# def update_entity(entities, entity, label, DEBUG=False):
#   entity_name = join_strings_smartly(entity)
#   entities.append((entity_name, label))
#   if DEBUG:
#     print(f"named entity: {entity_name} --> {label}")
#   return []
#
#
# def tag_mapping(word):
#   if word in ["Lumpur", "Pietroburgo", "del", "Capo"]:
#     return "I"
#   return "B"
#
#
# def add_to_json(output_jsonl, pandas_df, dataset_splits, dsplit, check_dtype, DEBUG=False):
#   """This function select the elements of a pandas dataframe that represent a named entity, (i.e., the words tagged with B-xxx, I-xxx) and add them to a given jsonl dataset."""
#
#   labels_map = {
#     "PER": 0,
#     "ORG": 1,
#     "LOC": 2
#   }
#   previous = {
#     "tag": np.nan, # B, C, O
#     "word": np.nan, # word or symbol
#     "label": np.nan # PER, ORG, LOC
#   }
#   entities = []
#   entity = []
#   sentence = []
#   sentence_counter = 0
#   map_tag = False
#
#   if check_dtype:
#     # reset sentence indices
#     dataset_splits[dsplit] = 0
#
#   for i, data in enumerate(df.itertuples()): 
#
#     # keep track of the next item in the dataset
#     if (i + 1) < len(df):
#       next = df.iloc[i + 1] 
#       if pd.isna(next.label):
#         next_tag = "stop"
#       else:
#         next_tag = next.label[0]
#
#     # update sentence counter
#     sentence_id = dataset_splits[dsplit]
#     allow_change = True
#
#     # condictions for blank line or middle stop
#     if (pd.isna(data.word) and pd.isna(data.label)) or (data.word == "." and not pd.isna(next.word)) or\
#       (data.word == ";" and not pd.isna(next.word)):
#
#       # CORRECT BAD FORMATTED SAMPLES FOR TASK WN TEST
#       if output_json == "NERMuD_WN_test.jsonl":
#         if previous["word"] == "Laura" and next.word == "Vasco":
#           next_tag = "O"
#         if next.word in ["Baku", "Delhi"]:
#           allow_change = False
#           map_tag = True
#         if previous["word"] in ["Dubai", "Budapest"]:
#           map_tag = False
#
#       if previous["tag"] in ["B", "I"] and \
#         (next_tag != "I"): # this is veeery specific :<
#         # save previous entity
#         entity = update_entity(entities, entity, label=labels_map[previous["label"]], DEBUG=DEBUG)
#
#         # reset values for previus sample
#         previous["tag"] = np.nan
#         previous["word"] = np.nan
#         previous["label"] = np.nan
#
#       if False:
#         """This control is made just identify the bad formatted sentences.
#         Those sentences are then deal with case by case since no clear pattern appear!!!"""
#
#         if previous["tag"] in ["B", "I"] and (next_tag == "I"):
#           if previous["tag"] == "B" : print("-"*100 + "\n")
#           print(f"Error sequence: {previous['word']} {previous['tag']}_{previous['label']} ---EMPTY LINE--- {next.word} {next.label}")
#           print()
#
#       # change sentence
#       break_symbols = {',', ':', '!', '?','<','(', '[', '{', '}', ']', ')', '>'}
#       if (previous["word"] not in break_symbols) and \
#         (next_tag != "I") and allow_change: # also this is veeery specific :<
#         # make json objects with entities
#         if len(entities) > 0:
#           sentence_text = join_strings_smartly(sentence)
#           make_json(output_jsonl, sentence_id, sentence_text, entities, DEBUG=DEBUG)
#           # update sentence index
#           dataset_splits[dsplit] += 1
#
#         if DEBUG:
#           print("\nSentence:")
#           print(sentence_text)
#           print("\nEntities:")
#           print(entities)
#           print("\n" + "-"*100 + "\n")
#
#         sentence = [] # delete old sentence
#         entities = [] # delete old entities
#
#         # reset values for previus sample
#         previous["tag"] = np.nan
#         previous["word"] = np.nan
#         previous["label"] = np.nan
#
#         # just for statistics
#         sentence_counter += 1
#
#       # same sentence
#       else:
#         ... # do nothing, somethimes is important to do nothing and just relax  
#
#     else:
#       tag = data.label[0]
#       word = data.word
#       label = data.label[2:]
#
#       if output_json == "NERMuD_WN_test.jsonl":
#         if data.word == "Vasco" and pd.isna(previous["word"]):
#           tag = "B"
#         if map_tag:
#           tag = tag_mapping(word)
#
#       if (tag == "O" and next_tag == "I"):
#         print(f"Empty identity: {word}")
#
#       # condictions for non-blank lines
#       if (tag == "B" and previous["tag"] in ["B", "I"]):
#         # save previous entity
#         entity = update_entity(entities, entity, label=labels_map[previous["label"]], DEBUG=DEBUG)
#         # go on 
#         entity.append(data.word)
#
#       elif (tag == "O"  and previous["tag"] in ["B", "I"]):
#         # save previous
#         entity = update_entity(entities, entity, label=labels_map[previous["label"]], DEBUG=DEBUG)
#
#       elif (tag == "B" and previous["tag"] not in ["B", "I"]) or \
#           (tag == "I" and previous["tag"] in ["B", "I"]) :
#         # go on
#         entity.append(data.word) 
#
#       else:
#         ... # do nothing
#
#       # update values of previous sample
#       previous["tag"] = tag
#       previous["word"] = word
#       previous["label"] = label
#
#       # and add words/symbols in the sentence
#       sentence.append(data.word)
#
#     if DEBUG:
#       print(data.Index, end=" ")
#       print(data.word, end=" ") 
#       print(data.label)
#       print(f"next label --> {next.label}")
#
#   return sentence_counter
#
#
# # MAIN
# if __name__ == '__main__' : 
#
#   # set up command line args
#   parser = argparse.ArgumentParser(description='Dataset Manipulation')
#   parser.add_argument('--debug', '-d', action='store_true')
#   parser.add_argument('--single', '-s', action='store_true')
#   args = parser.parse_args()
#   DEBUG = args.debug 
#   SINGLE = args.single
#
#   # global variables
#   results_dir = "./results/"
#   data_dir = "./data/"
#   os.makedirs(data_dir, exist_ok=True)
#   dataset_splits = {
#     "train": 0,
#     "test": 0,
#     "dev": 0,
#   }
#   dataset_types = ["ADG", "FIC", "WN"]
#   jsonl_files = set()
#
#   # get datasets
#   for dtype in dataset_types:
#     for dsplit in  dataset_splits.keys():
#       dataset = dtype + "_" + dsplit +".tsv"
#       if not os.path.exists(data_dir + dataset):
#         data_url = "https://github.com/dhfbk/KIND/raw/main/evalita-2023/" + dataset
#         wget.download(url=data_url, out=data_dir+dataset, bar=progress_bar)    
#         print()
#       else:
#         print(f"Dataset: {dataset} already present")
#         ...
#
#   # clear previous results just to be sure
#   try:
#     shutil.rmtree(results_dir)
#   except FileNotFoundError:
#     pass
#
#   # make json files
#   print("Working...")
#   for dtype in dataset_types:
#     for dsplit in  dataset_splits.keys():
#       dataset = dtype + "_" + dsplit +".tsv"
#       # save multiple jsonl files for each dataset 
#       if SINGLE:
#         output_json = "NERMuD_" + dsplit + ".jsonl"
#         check_dtype = False
#       # or save just one file for each split
#       else:
#         output_json = "NERMuD_" + dtype + "_" + dsplit + ".jsonl"
#         check_dtype = True
#
#       print(f"Dataset --> {dataset}")
#       df = make_dataframe(data_dir + dataset, dtype)
#       print(f"Dataset len: {len(df)}")
#       total_sentence = add_to_json(output_json, df, dataset_splits, dsplit, check_dtype, DEBUG=DEBUG)
#       jsonl_files.add(output_json)
#       print(f"JSONL output --> {output_json}")
#       print(f"JSONL sentences --> {total_sentence}")
#
#   print(jsonl_files)
#   move_data(list(jsonl_files), results_dir)
