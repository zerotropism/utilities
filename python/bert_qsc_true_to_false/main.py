import argparse
import logging
import os

from modules.true_to_false import true_to_false
from modules.split import split
from modules.concat_and_shuffle import concat_and_shuffle

logFileName = "run_log.txt"

# vérifie l'existence du fichier log
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

# récupérer l'input utilisateur
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", help="directory where data located", type=str
)
parser.add_argument("--tsv_in_true", help="dataset to load", type=str)
parser.add_argument("--tsv_out_false", help="out dataset", type=str)
args = parser.parse_args()

# appel des méthodes
true_to_false(args.data_dir, args.tsv_in_true, args.tsv_out_false)
(tsv_in_true_train, tsv_in_true_test) = split(
    args.data_dir, args.tsv_in_true, 0.2
)
(tsv_out_false_train, tsv_out_false_test) = split(
    args.data_dir, args.tsv_out_false, 0.2
)

concat_and_shuffle(args.data_dir, tsv_in_true_train, tsv_out_false_train)
concat_and_shuffle(args.data_dir, tsv_in_true_test, tsv_out_false_test)
