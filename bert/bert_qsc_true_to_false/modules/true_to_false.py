import pandas as pd
import numpy as np
import csv
import os.path


def true_to_false(data_dir, tsv_in_file, tsv_out_file):

    data_in = pd.read_csv(data_dir + "/" + tsv_in_file, delimiter="\t")
    print(data_in.head())

    data_out = {}
    data_out["sentence"] = []
    data_out["label"] = []

    tmp_dict = {}
    tmp_dict["question"] = []
    tmp_dict["context"] = []

    for i in range(len(data_in["sentence"])):
        tmp = data_in["sentence"][i].split("[SEP]")
        tmp_dict["question"].append(tmp[0])
        tmp_dict["context"].append(tmp[1])

    for i in range(len(tmp_dict["context"])):
        for j in range(len(tmp_dict["context"]) - i):
            if tmp_dict["context"][i] != tmp_dict["context"][j + i]:
                sentence = (
                    tmp_dict["question"][i]
                    + "[SEP]"
                    + tmp_dict["context"][j + i]
                )
                data_out["sentence"].append(sentence)
                data_out["label"].append("1")
                break

    prefinal = np.column_stack((data_out["sentence"], data_out["label"]))
    final = pd.DataFrame(prefinal, columns=["sentence", "label"])

    final.to_csv(
        data_dir + "/" + tsv_out_file,
        sep="\t",
        encoding="utf-8",
        index=False,
        quoting=csv.QUOTE_NONE,
    )

    if os.path.exists(data_dir + "/" + tsv_out_file):
        print(
            "File {} CORRECTLY generated!".format(
                data_dir + "/" + tsv_out_file
            )
        )
    else:
        print("File {} NOT generated!".format(data_dir + "/" + tsv_out_file))

    return 0
