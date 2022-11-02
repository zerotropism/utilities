import argparse
import json
import pandas as pd
import logging
import csv

logFileName = "run_log2.txt"

import os

if os.path.exists(logFileName):
    print("File {} exists, removing it".format(logFileName))
    os.remove(logFileName)
else:
    print("File {} does not exist, creating new one".format(logFileName))
logging.basicConfig(
    filename=logFileName,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)

from modules.to_QSC import create_dataset_QSC
from modules.to_QSC import split_dataset

# récupère l'input utilisateur
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_json_in", help='train JSON file to convert"', type=str
)
parser.add_argument(
    "--dev_json_in", help='dev JSON file to convert"', type=str
)
parser.add_argument(
    "--BERT_task", help='type of task for BERT, "QSC"', type=str
)
parser.add_argument(
    "--train_test_ration",
    help='ratio to split the given dataset, example "0.2" for "80 in train" and "20 in test"',
    type=float,
)
args = parser.parse_args()

if args.BERT_task == "QSC":
    tsv_final_train = create_dataset_QSC(args.train_json_in, args.dev_json_in)

    print("tsv_final_train: {} ".format(tsv_final_train))

    split_dataset(tsv_final_train, args.train_test_ration)
