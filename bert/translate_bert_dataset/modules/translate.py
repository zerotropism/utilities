# from translate import Translator
import os
import sys
import argparse
import torch
import fairseq

# List available models
torch.hub.list("pytorch/fairseq")  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
# en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')

# Load a transformer trained on WMT'14 En-Fr
en2fr = torch.hub.load(
    "pytorch/fairseq",
    "transformer.wmt14.en-fr",
    tokenizer="moses",
    bpe="subword_nmt",
)


# The underlying model is available under the *models* attribute
assert isinstance(en2fr.models[0], fairseq.models.transformer.TransformerModel)


def translate(file):

    filepath = os.path.basename(file)
    filename_wo_ext = os.path.splitext(filepath)[0]

    lineList = []

    try:
        file_read = open(file)
        lineList = file_read.readlines()
    finally:
        file_read.close()

    file_write = open(filename_wo_ext + "_fr.txt", "w+")
    for i in range(len(lineList)):

        current_translate = en2fr.translate(lineList[i])

        # print('line {} EN: {}'.format(i, lineList[i]))
        # print('line {} FR: {}'.format(i, current_translate))

        file_write.write(current_translate + "\n")

        # if(i==10):
        #    break
    file_write.close()


# getting user inputs
parser = argparse.ArgumentParser()
parser.add_argument(
    "--language2language", help='en2fr"', type=str, default="NA"
)
parser.add_argument(
    "--text2translate",
    help='text to translate in specified language"',
    type=str,
    default="NA",
)
args = parser.parse_args()

if args.language2language == "en2fr":
    current_translate = en2fr.translate(args.text2translate)
    print(current_translate)
