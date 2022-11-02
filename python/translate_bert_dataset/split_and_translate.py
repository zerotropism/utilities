import argparse
import json
import pandas as pd
import logging
import csv
import datetime

logFileName = "run_log.txt"

# vérifie que le log file existe
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

from modules.split_questions_sequences import create_text_to_translate
from modules.translate import translate


# récupère l'input utilisateur
parser = argparse.ArgumentParser()
parser.add_argument("--tsv_in", help="TSV file input", type=str)
parser.add_argument(
    "--lines_per_generated_file",
    help="ouput dir where to generate translated files",
    type=int,
)
args = parser.parse_args()

print(
    "splitting file {} - started at {}".format(
        args.tsv_in, datetime.datetime.now()
    )
)
(file_name_questions, file_name_sentences) = create_text_to_translate(
    args.tsv_in, "en", args.lines_per_generated_file
)
print(
    "splitting file {} - finished at {}".format(
        args.tsv_in, datetime.datetime.now()
    )
)

count_lines = len(open(file_name_questions).readlines())
print(
    "number of lines in file {}: {}".format(file_name_questions, count_lines)
)
count_lines = len(open(file_name_sentences).readlines())
print(
    "number of lines in file {}: {}".format(file_name_sentences, count_lines)
)

print(
    "translating {} - started at {}".format(
        file_name_questions, datetime.datetime.now()
    )
)
translate(file_name_questions)
print(
    "translating {} - finished at {}".format(
        file_name_questions, datetime.datetime.now()
    )
)

print(
    "translating {} - started at {}".format(
        file_name_sentences, datetime.datetime.now()
    )
)
translate(file_name_sentences)
print(
    "translating {} - finished at {}".format(
        file_name_sentences, datetime.datetime.now()
    )
)
