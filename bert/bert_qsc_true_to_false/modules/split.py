import os
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from shutil import copyfile


def split(data_dir, tsv_in, split_ratio):

    base = os.path.basename(data_dir + "/" + tsv_in)

    (filename, extention) = os.path.splitext(base)
    # print('Dataset to split: {} '.format(filename))

    data = pd.read_csv(
        data_dir + "/" + tsv_in, delimiter="\t", encoding="utf-8"
    )
    # print('Head 10 of full dataset')
    # print(data.head(10))
    # print('-----------------------')

    train, test = train_test_split(data, test_size=split_ratio)

    tsv_train = data_dir + "/" + filename + "_train.tsv"
    train.to_csv(
        tsv_train,
        sep="\t",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_NONE,
    )
    if os.path.isfile(tsv_train):
        print("File {} CORRECTLY generated".format(tsv_train))
        # logging.debug('File {} correctly generated'.format(tsv))
    else:
        print("File {} NOT generated !".format(tsv_train))
        # logging.debug('File {} NOT generated !'.format(tsv))

    tsv_test = data_dir + "/" + filename + "_test.tsv"
    test.to_csv(
        tsv_test,
        sep="\t",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_NONE,
    )
    if os.path.isfile(tsv_test):
        print("File {} CORERCTLY generated".format(data_dir + "/" + tsv_test))
        # logging.debug('File {} correctly generated'.format(tsv))
    else:
        print("File {} NOT generated !".format(data_dir + "/" + tsv_test))
        # logging.debug('File {} NOT generated !'.format(tsv))

    # print('Head 10 of train dataset')
    # print(train.head(10))
    # print('-----------------------')
    # logging.debug('Head 10 of train dataset')
    # logging.debug(train.head(10))
    # logging.debug('-----------------------')

    # print('Head 10 of test dataset')
    # print(test.head(10))
    # print('-----------------------')
    # logging.debug('Head 10 of test dataset')
    # logging.debug(test.head(10))
    # logging.debug('-----------------------')

    # print('Number of rows in full dataset: {} '.format(len(data)))
    # print('Number of rows in train dataset: {} '.format(len(train)))
    # print('Number of rows in test dataset: {} '.format(len(test)))
    # print('Sum train + test : {} '.format(len(train) + len(test)))
    # logging.debug('Number of rows in full dataset: {} '.format(len(data)))
    # logging.debug('Number of rows in train dataset: {} '.format(len(train)))
    # logging.debug('Number of rows in test dataset: {} '.format(len(test)))
    # logging.debug('Sum train + test : {} '.format(len(train) + len(test)))

    if (len(train) + len(test)) == len(data):
        print("data CORRECTLY splitted !")
    else:
        print("data NOT correctly splitted !")

    return (tsv_train, tsv_test)
