import os
import pandas as pd
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from shutil import copyfile


def concat_and_shuffle(data_dir, tsv_true, tsv_false):

    tsv_true_file = tsv_true
    tsv_false_file = tsv_false

    data_tsv_true = pd.read_csv(tsv_true_file, delimiter="\t")
    data_tsv_false = pd.read_csv(tsv_false_file, delimiter="\t")

    base = os.path.basename(tsv_true)

    (filename, extention) = os.path.splitext(base)

    data_concat = pd.concat([data_tsv_true, data_tsv_false])

    data_shuffle = shuffle(data_concat)

    tsv_concat_shuffle = filename + "_false_concat_shuffle.tsv"

    data_shuffle.to_csv(
        data_dir + "/" + tsv_concat_shuffle,
        sep="\t",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_NONE,
    )

    if (len(data_tsv_true) + len(data_tsv_false)) == len(data_shuffle):
        print("data shuffle CORRECTLY splitted !")
    else:
        print("data shuffle NOT correctly splitted !")

    return tsv_concat_shuffle
