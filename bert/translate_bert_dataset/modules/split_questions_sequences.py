import json
import logging
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import csv
from shutil import copyfile
import re


def load_tsv(tsv_in):
    df = pd.read_csv(tsv_in, sep="\t", header=False)

    return df


def create_text_to_translate(tsv_in, language, line_modulo):

    filepath = os.path.basename(tsv_in)
    filename_wo_ext = os.path.splitext(filepath)[0]

    df = pd.read_csv(tsv_in, sep="\t")

    sorting = True
    questions_list = []
    sentences_list = []
    for i in range(len(df["sentence"])):
        questions_sentences = df["sentence"][i]

        start_q = "[CLS]"
        end_q = "[SEP]"
        question = questions_sentences[
            questions_sentences.find(start_q)
            + len(start_q) : questions_sentences.rfind(end_q)
        ]
        questions_list.append(question + "\n")

        start_s = "[SEP]"
        end_s = " \n"
        sentence = questions_sentences[
            questions_sentences.find(start_s)
            + len(start_s) : questions_sentences.rfind(end_s)
        ]
        sentences_list.append(sentence + ".\n")

    outer_count = 1
    line_count = 0
    while sorting:
        count = 0
        increment = (outer_count - 1) * line_modulo
        left = len(questions_list) - increment

        file_name_questions = (
            filename_wo_ext
            + "_questions_"
            + language
            + "_"
            + str(outer_count * line_modulo)
            + ".txt"
        )
        file_name_sentences = (
            filename_wo_ext
            + "_sentences_"
            + language
            + "_"
            + str(outer_count * line_modulo)
            + ".txt"
        )

        hold_new_lines_q = []
        hold_new_lines_s = []

        if left < line_modulo:
            while count < left:
                hold_new_lines_q.append(questions_list[line_count])
                hold_new_lines_s.append(sentences_list[line_count])
                count += 1
                line_count += 1
            sorting = False
        else:
            while count < line_modulo:
                hold_new_lines_q.append(questions_list[line_count])
                hold_new_lines_s.append(sentences_list[line_count])
                count += 1
                line_count += 1
        outer_count += 1

        with open(file_name_questions, "w", encoding="UTF-8") as file_q:
            for row in hold_new_lines_q:
                file_q.write(row)
        file_q.closed

        with open(file_name_sentences, "w", encoding="UTF-8") as file_s:
            for row in hold_new_lines_s:
                file_s.write(row)
        file_s.closed

    return (file_name_questions, file_name_sentences)


def build_true_questions_for_train_dev_df(json_in):
    data_dict = {}
    data_dict["sentence"] = []
    data_dict["label"] = []

    json_dict = read_json(json_in)

    logging.debug(" - main: length data: {}".format(len(json_dict["data"])))
    for i in range(len(json_dict["data"])):
        logging.debug(
            " - main: ----title[{}]: {}".format(
                i, (json_dict["data"][i]["title"])
            )
        )

        for j in range(len(json_dict["data"][i]["paragraphs"])):
            logging.debug(
                " - main: --------context[{}]: {}".format(
                    j, (json_dict["data"][i]["paragraphs"][j]["context"])
                )
            )

            current_context = json_dict["data"][i]["paragraphs"][j]["context"]
            current_context_sub = current_context.replace("\n", " ")

            for k in range(len(json_dict["data"][i]["paragraphs"][j]["qas"])):
                current_sentence = (
                    "[CLS] "
                    + json_dict["data"][i]["paragraphs"][j]["qas"][k][
                        "question"
                    ]
                    + " [SEP] "
                    + current_context_sub
                )
                data_dict["sentence"].append(current_sentence)
                data_dict["label"].append("0")

                logging.debug(
                    " - main: ------------sentence[{}]: {}".format(
                        k, current_sentence
                    )
                )
                logging.debug(
                    " - main: ------------label[{}]: {}".format(k, "0")
                )

    df = pd.DataFrame(data_dict)

    return df


def build_false_questions_for_train_dev_df(json_in):
    data_dict = {}
    data_dict["sentence"] = []
    data_dict["label"] = []

    json_dict = read_json(json_in)

    logging.debug(" - main: length data: {}".format(len(json_dict["data"])))
    for i in range(len(json_dict["data"])):
        logging.debug(
            " - main: ----title[{}]: {}".format(
                i, (json_dict["data"][i]["title"])
            )
        )

        dict_paragraphs = {}
        dict_paragraphs["context"] = []

        for j in range(len(json_dict["data"][i]["paragraphs"])):
            logging.debug(
                " - main: --------context[{}]: {}".format(
                    j, (json_dict["data"][i]["paragraphs"][j]["context"])
                )
            )
            dict_paragraphs["context"].append(
                json_dict["data"][i]["paragraphs"][j]["context"]
            )

        df_contexts = pd.DataFrame(dict_paragraphs)

        for j in range(len(json_dict["data"][i]["paragraphs"])):
            logging.debug(
                " - main: --------context[{}]: {}".format(
                    j, (json_dict["data"][i]["paragraphs"][j]["context"])
                )
            )

            df_contexts_random = df_contexts.sample(n=1)
            contexts_random_index = df_contexts_random["context"].index

            contexts_random = df_contexts["context"][contexts_random_index[0]]
            contexts_random_sub = contexts_random.replace("\n", " ")

            if contexts_random_index[0] != j:

                logging.debug(
                    " - main: --------randomly selected context: {}".format(
                        contexts_random_sub
                    )
                )

                for k in range(
                    len(json_dict["data"][i]["paragraphs"][j]["qas"])
                ):
                    current_sentence = (
                        "[CLS] "
                        + json_dict["data"][i]["paragraphs"][j]["qas"][k][
                            "question"
                        ]
                        + " [SEP] "
                        + contexts_random_sub
                    )

                    data_dict["sentence"].append(current_sentence)
                    data_dict["label"].append("1")

                    logging.debug(
                        " - main: ------------sentence[{}]: {}".format(
                            k, current_sentence
                        )
                    )
                    logging.debug(
                        " - main: ------------label[{}]: {}".format(k, "1")
                    )

    df = pd.DataFrame(data_dict)

    return df


def build_questions_for_test_df(json_in):
    data_dict = {}
    data_dict["index"] = []
    data_dict["sentence"] = []

    json_dict = read_json(json_in)

    index = 0

    logging.debug(" - main: length data: {}".format(len(json_dict["data"])))
    for i in range(len(json_dict["data"])):
        logging.debug(
            " - main: ----title[{}]: {}".format(
                i, (json_dict["data"][i]["title"])
            )
        )

        for j in range(len(json_dict["data"][i]["paragraphs"])):
            logging.debug(
                " - main: --------context[{}]: {}".format(
                    j, (json_dict["data"][i]["paragraphs"][j]["context"])
                )
            )

            for k in range(len(json_dict["data"][i]["paragraphs"][j]["qas"])):
                current_sentence = json_dict["data"][i]["paragraphs"][j][
                    "qas"
                ][k]["question"]
                data_dict["index"].append(index)
                data_dict["sentence"].append(current_sentence)

                index = index + 1

                logging.debug(
                    " - main: ------------sentence[{}]: {}".format(
                        k, current_sentence
                    )
                )

    df = pd.DataFrame(data_dict)

    return df


def create_dataset_SST_2(train_json_in, dev_json_in):

    base = os.path.basename(train_json_in)

    (filename, extention) = os.path.splitext(base)
    print("Filename for train dataset: {} ".format(filename))
    logging.debug("Filename for train dataset: {} ".format(filename))

    df_true = build_true_questions_for_train_dev_df(train_json_in)
    tsv = filename + "_SST_2_true.tsv"
    df_true.to_csv(
        tsv, sep="\t", encoding="utf-8", index=False, quoting=csv.QUOTE_NONE
    )
    if os.path.isfile(tsv):
        print("File {} correctly generated".format(tsv))
        logging.debug("File {} correctly generated".format(tsv))
    else:
        print("File {} NOT generated !".format(tsv))
        logging.debug("File {} NOT generated !".format(tsv))

    df_false = build_false_questions_for_train_dev_df(train_json_in)
    tsv = filename + "_SST_2_false.tsv"
    df_false.to_csv(
        tsv, sep="\t", encoding="utf-8", index=False, quoting=csv.QUOTE_NONE
    )
    if os.path.isfile(tsv):
        print("File {} correctly generated".format(tsv))
        logging.debug("File {} correctly generated".format(tsv))
    else:
        print("File {} NOT generated !".format(tsv))
        logging.debug("File {} NOT generated !".format(tsv))

    df_concat = pd.concat([df_true, df_false])
    tsv = filename + "_SST_2_concat.tsv"
    df_concat.to_csv(
        tsv, sep="\t", encoding="utf-8", index=False, quoting=csv.QUOTE_NONE
    )
    if os.path.isfile(tsv):
        print("File {} correctly generated".format(tsv))
        logging.debug("File {} correctly generated".format(tsv))
    else:
        print("File {} NOT generated !".format(tsv))
        logging.debug("File {} NOT generated !".format(tsv))

    df_shuffle = shuffle(df_concat)
    tsv_final_train = filename + "_SST_2_train.tsv"
    df_shuffle.to_csv(
        tsv_final_train,
        sep="\t",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_NONE,
    )
    if os.path.isfile(tsv_final_train):
        print("File {} correctly generated".format(tsv_final_train))
        logging.debug("File {} correctly generated".format(tsv_final_train))
    else:
        print("File {} NOT generated !".format(tsv_final_train))
        logging.debug("File {} NOT generated !".format(tsv_final_train))

    base = os.path.basename(dev_json_in)

    (filename, extention) = os.path.splitext(base)
    print("Filename for dev dataset: {} ".format(filename))
    logging.debug("Filename for dev dataset: {} ".format(filename))

    df_dev_for_test = build_questions_for_test_df(dev_json_in)
    tsv_dev_final = filename + "_SST_2_test.tsv"
    df_dev_for_test.to_csv(
        tsv_dev_final,
        sep="\t",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_NONE,
    )
    if os.path.isfile(tsv_dev_final):
        print("File {} correctly generated".format(tsv_dev_final))
        logging.debug("File {} correctly generated".format(tsv_dev_final))
    else:
        print("File {} NOT generated !".format(tsv_dev_final))
        logging.debug("File {} NOT generated !".format(tsv_dev_final))

    copyfile(tsv_dev_final, "test.tsv")
    if os.path.isfile("test.tsv"):
        print("File {} correctly generated".format("test.tsv"))
        logging.debug("File {} correctly generated".format("test.tsv"))
    else:
        print("File {} NOT generated !".format("test.tsv"))
        logging.debug("File {} NOT generated !".format("test.tsv"))

    return tsv_final_train


def split_dataset(train_tsv_in, split_ratio):
    dataset = train_tsv_in

    base = os.path.basename(dataset)

    (filename, extention) = os.path.splitext(base)
    print("Dataset to split: {} ".format(filename))

    df = pd.read_csv(dataset, delimiter="\t", encoding="utf-8")
    print("Head 10 of full dataset")
    print(df.head(10))
    print("-----------------------")

    train, test = train_test_split(df, test_size=split_ratio)
    tsv = filename + "_train.tsv"
    train.to_csv(
        tsv, sep="\t", encoding="utf-8", index=False, quoting=csv.QUOTE_NONE
    )
    if os.path.isfile(tsv):
        print("File {} correctly generated".format(tsv))
        logging.debug("File {} correctly generated".format(tsv))
    else:
        print("File {} NOT generated !".format(tsv))
        logging.debug("File {} NOT generated !".format(tsv))

    copyfile(tsv, "train.tsv")
    if os.path.isfile("train.tsv"):
        print("File {} correctly generated".format("train.tsv"))
        logging.debug("File {} correctly generated".format("train.tsv"))
    else:
        print("File {} NOT generated !".format("train.tsv"))
        logging.debug("File {} NOT generated !".format("train.tsv"))

    tsv = filename + "_test.tsv"
    test.to_csv(
        tsv, sep="\t", encoding="utf-8", index=False, quoting=csv.QUOTE_NONE
    )
    if os.path.isfile(tsv):
        print("File {} correctly generated".format(tsv))
        logging.debug("File {} correctly generated".format(tsv))
    else:
        print("File {} NOT generated !".format(tsv))
        logging.debug("File {} NOT generated !".format(tsv))

    copyfile(tsv, "dev.tsv")
    if os.path.isfile("dev.tsv"):
        print("File {} correctly generated".format("dev.tsv"))
        logging.debug("File {} correctly generated".format("dev.tsv"))
    else:
        print("File {} NOT generated !".format("dev.tsv"))
        logging.debug("File {} NOT generated !".format("dev.tsv"))

    print("Head 10 of train dataset")
    print(train.head(10))
    print("-----------------------")
    logging.debug("Head 10 of train dataset")
    logging.debug(train.head(10))
    logging.debug("-----------------------")

    print("Head 10 of test dataset")
    print(test.head(10))
    print("-----------------------")
    logging.debug("Head 10 of test dataset")
    logging.debug(test.head(10))
    logging.debug("-----------------------")

    print("Number of rows in full dataset: {} ".format(len(df)))
    print("Number of rows in train dataset: {} ".format(len(train)))
    print("Number of rows in test dataset: {} ".format(len(test)))
    print("Sum train + test : {} ".format(len(train) + len(test)))
    logging.debug("Number of rows in full dataset: {} ".format(len(df)))
    logging.debug("Number of rows in train dataset: {} ".format(len(train)))
    logging.debug("Number of rows in test dataset: {} ".format(len(test)))
    logging.debug("Sum train + test : {} ".format(len(train) + len(test)))

    return 0


def test_sub():

    str_origin = "Allied success against Japan.\nAfter the Doolittle Raid, the Japanese army conducted"
    print("str_origin: {} ".format(str_origin))

    str_sub = str_origin.replace("\n", " ")
    print("str_sub: {} ".format(str_sub))
