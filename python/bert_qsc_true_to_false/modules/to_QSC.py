import json
import logging
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import csv
from shutil import copyfile

import torch
import fairseq

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
#en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')

# Load a transformer trained on WMT'14 En-Fr
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')


# The underlying model is available under the *models* attribute
assert isinstance(en2fr.models[0], fairseq.models.transformer.TransformerModel)


def read_json(json_in):
    json_dict = {}

    with open(json_in, 'r') as f:
        json_dict = json.load(f)

    return json_dict


def build_true_questions_for_train_dev_df(json_in):
    data_dict = {}
    data_dict['sentence'] = []
    data_dict['label'] = []

    json_dict = read_json(json_in)
    
    logging.debug(" - main: length data: {}".format(len(json_dict['data'])))
    for i in range(len(json_dict['data'])):
        logging.debug(" - main: ----title[{}]: {}".format(i, (json_dict['data'][i]['title'])))

        for j in range(len(json_dict['data'][i]['paragraphs'])):
            logging.debug(" - main: --------context[{}]: {}".format(j, (json_dict['data'][i]['paragraphs'][j]['context'])))

            current_context = json_dict['data'][i]['paragraphs'][j]['context']
            current_context_sub_en = current_context.replace("\n", " ")
            current_context_sub_fr = en2fr.translate(current_context_sub_en)
   
            for k in range(len(json_dict['data'][i]['paragraphs'][j]['qas'])):
                
                current_question_en = json_dict['data'][i]['paragraphs'][j]['qas'][k]['question']
                current_question_fr = en2fr.translate(current_question_en)

                current_sentence_en = '[CLS] ' + current_question_en + ' [SEP] ' + current_context_sub_en
                current_sentence_fr = '[CLS] ' + current_question_fr + ' [SEP] ' + current_context_sub_fr

                data_dict['sentence'].append(current_sentence_en)
                data_dict['sentence'].append(current_sentence_fr)
                data_dict['label'].append('0')
                data_dict['label'].append('0')

                logging.debug(" - main: ------------sentence_en[{}]: {}".format(k, current_sentence_en))
                logging.debug(" - main: ------------sentence_fr[{}]: {}".format(k, current_sentence_fr))
                logging.debug(" - main: ------------label[{}]: {}".format(k, '0'))

        #if(i==2):
        #    break

    df = pd.DataFrame(data_dict)

    return df



def build_false_questions_for_train_dev_df(json_in):
    data_dict = {}
    data_dict['sentence'] = []
    data_dict['label'] = []
    
    json_dict = read_json(json_in)
    
    logging.debug(" - main: length data: {}".format(len(json_dict['data'])))
    for i in range(len(json_dict['data'])):
        logging.debug(" - main: ----title[{}]: {}".format(i, (json_dict['data'][i]['title'])))
        
        dict_paragraphs = {}
        dict_paragraphs['context'] = []

        for j in range(len(json_dict['data'][i]['paragraphs'])):
            logging.debug(" - main: --------context[{}]: {}".format(j, (json_dict['data'][i]['paragraphs'][j]['context'])))
            dict_paragraphs['context'].append(json_dict['data'][i]['paragraphs'][j]['context'])

        df_contexts = pd.DataFrame(dict_paragraphs)
        
        for j in range(len(json_dict['data'][i]['paragraphs'])):
            logging.debug(" - main: --------context[{}]: {}".format(j, (json_dict['data'][i]['paragraphs'][j]['context'])))

            df_contexts_random = df_contexts.sample(n=1)
            contexts_random_index = df_contexts_random['context'].index
            
            contexts_random = df_contexts['context'][contexts_random_index[0]]
            contexts_random_sub_en = contexts_random.replace("\n", " ")
            contexts_random_sub_fr = en2fr.translate(contexts_random_sub_en)
        
            if(contexts_random_index[0] != j):

                logging.debug(" - main: --------randomly selected context: {}".format(contexts_random_sub_en))

                for k in range(len(json_dict['data'][i]['paragraphs'][j]['qas'])):
                    current_question_en = json_dict['data'][i]['paragraphs'][j]['qas'][k]['question']
                    current_question_fr = en2fr.translate(current_question_en)

                    current_sentence_en = '[CLS] ' + current_question_en + ' [SEP] ' + contexts_random_sub_en
                    current_sentence_fr = '[CLS] ' + current_question_fr + ' [SEP] ' + contexts_random_sub_fr
            
                    data_dict['sentence'].append(current_sentence_en)
                    data_dict['sentence'].append(current_sentence_fr)                    
                    data_dict['label'].append('1')
                    data_dict['label'].append('1')

                    logging.debug(" - main: ------------sentence_en[{}]: {}".format(k, current_sentence_en))
                    logging.debug(" - main: ------------sentence_fr[{}]: {}".format(k, current_sentence_fr))
                    logging.debug(" - main: ------------label[{}]: {}".format(k, '1'))


        #if(i==2):
        #    break
    
    df = pd.DataFrame(data_dict)

    return df



def build_questions_for_test_df(json_in):
    data_dict = {}
    data_dict['index'] = []
    data_dict['sentence'] = []

    json_dict = read_json(json_in)
    
    index=0

    logging.debug(" - main: length data: {}".format(len(json_dict['data'])))
    for i in range(len(json_dict['data'])):
        logging.debug(" - main: ----title[{}]: {}".format(i, (json_dict['data'][i]['title'])))

        for j in range(len(json_dict['data'][i]['paragraphs'])):
            logging.debug(" - main: --------context[{}]: {}".format(j, (json_dict['data'][i]['paragraphs'][j]['context'])))

            for k in range(len(json_dict['data'][i]['paragraphs'][j]['qas'])):
                current_sentence_en = json_dict['data'][i]['paragraphs'][j]['qas'][k]['question']
                data_dict['index'].append(index)
                data_dict['sentence'].append(current_sentence_en)
                index = index + 1

                current_sentence_fr = en2fr.translate(current_sentence_en)
                data_dict['index'].append(index)
                data_dict['sentence'].append(current_sentence_fr)
                index = index + 1

                logging.debug(" - main: ------------sentence_en[{}]: {}".format(k, current_sentence_en))
                logging.debug(" - main: ------------sentence_fr[{}]: {}".format(k, current_sentence_fr))

        #if(i==2):
        #    break

    df = pd.DataFrame(data_dict)

    return df



def create_dataset_QSC(train_json_in, dev_json_in):

    base = os.path.basename(train_json_in)

    (filename, extention) = os.path.splitext(base)
    print('Filename for train dataset: {} '.format(filename))
    logging.debug('Filename for train dataset: {} '.format(filename))

    df_true = build_true_questions_for_train_dev_df(train_json_in)
    tsv = filename + '_QSC_true.tsv'
    df_true.to_csv(tsv, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv)):
        print('File {} correctly generated'.format(tsv))
        logging.debug('File {} correctly generated'.format(tsv))
    else:
        print('File {} NOT generated !'.format(tsv))
        logging.debug('File {} NOT generated !'.format(tsv))

    df_false = build_false_questions_for_train_dev_df(train_json_in)
    tsv = filename + '_QSC_false.tsv'
    df_false.to_csv(tsv, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv)):
        print('File {} correctly generated'.format(tsv))
        logging.debug('File {} correctly generated'.format(tsv))
    else:
        print('File {} NOT generated !'.format(tsv))
        logging.debug('File {} NOT generated !'.format(tsv))

    df_concat = pd.concat([df_true, df_false])
    tsv = filename + '_QSC_concat.tsv'
    df_concat.to_csv(tsv, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv)):
        print('File {} correctly generated'.format(tsv))
        logging.debug('File {} correctly generated'.format(tsv))
    else:
        print('File {} NOT generated !'.format(tsv))
        logging.debug('File {} NOT generated !'.format(tsv))

    df_shuffle = shuffle(df_concat)
    tsv_final_train = filename + '_QSC_train.tsv'
    df_shuffle.to_csv(tsv_final_train, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv_final_train)):
        print('File {} correctly generated'.format(tsv_final_train))
        logging.debug('File {} correctly generated'.format(tsv_final_train))
    else:
        print('File {} NOT generated !'.format(tsv_final_train))
        logging.debug('File {} NOT generated !'.format(tsv_final_train))



    base = os.path.basename(dev_json_in)

    (filename, extention) = os.path.splitext(base)
    print('Filename for dev dataset: {} '.format(filename))
    logging.debug('Filename for dev dataset: {} '.format(filename))

    df_dev_for_test = build_questions_for_test_df(dev_json_in)
    tsv_dev_final = filename + '_QSC_test.tsv'
    df_dev_for_test.to_csv(tsv_dev_final, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv_dev_final)):
        print('File {} correctly generated'.format(tsv_dev_final))
        logging.debug('File {} correctly generated'.format(tsv_dev_final))
    else:
        print('File {} NOT generated !'.format(tsv_dev_final))
        logging.debug('File {} NOT generated !'.format(tsv_dev_final))

    copyfile(tsv_dev_final, 'test.tsv')
    if(os.path.isfile('test.tsv')):
        print('File {} correctly generated'.format('test.tsv'))
        logging.debug('File {} correctly generated'.format('test.tsv'))
    else:
        print('File {} NOT generated !'.format('test.tsv'))
        logging.debug('File {} NOT generated !'.format('test.tsv'))

    return tsv_final_train



def split_dataset(train_tsv_in, split_ratio):
    dataset = train_tsv_in

    base = os.path.basename(dataset)

    (filename, extention) = os.path.splitext(base)
    print('Dataset to split: {} '.format(filename))

    df=pd.read_csv(dataset,delimiter='\t',encoding='utf-8')
    print('Head 10 of full dataset')
    print(df.head(10))
    print('-----------------------')

    train, test = train_test_split(df, test_size=split_ratio)
    tsv = filename + '_train.tsv'
    train.to_csv(tsv, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv)):
        print('File {} correctly generated'.format(tsv))
        logging.debug('File {} correctly generated'.format(tsv))
    else:
        print('File {} NOT generated !'.format(tsv))
        logging.debug('File {} NOT generated !'.format(tsv))
    
    copyfile(tsv, 'train.tsv')
    if(os.path.isfile('train.tsv')):
        print('File {} correctly generated'.format('train.tsv'))
        logging.debug('File {} correctly generated'.format('train.tsv'))
    else:
        print('File {} NOT generated !'.format('train.tsv'))
        logging.debug('File {} NOT generated !'.format('train.tsv'))

    tsv = filename + '_test.tsv'
    test.to_csv(tsv, sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    if(os.path.isfile(tsv)):
        print('File {} correctly generated'.format(tsv))
        logging.debug('File {} correctly generated'.format(tsv))
    else:
        print('File {} NOT generated !'.format(tsv))
        logging.debug('File {} NOT generated !'.format(tsv))

    copyfile(tsv, 'dev.tsv')
    if(os.path.isfile('dev.tsv')):
        print('File {} correctly generated'.format('dev.tsv'))
        logging.debug('File {} correctly generated'.format('dev.tsv'))
    else:
        print('File {} NOT generated !'.format('dev.tsv'))
        logging.debug('File {} NOT generated !'.format('dev.tsv'))

    print('Head 10 of train dataset')
    print(train.head(10))
    print('-----------------------')
    logging.debug('Head 10 of train dataset')
    logging.debug(train.head(10))
    logging.debug('-----------------------')

    print('Head 10 of test dataset')
    print(test.head(10))
    print('-----------------------')
    logging.debug('Head 10 of test dataset')
    logging.debug(test.head(10))
    logging.debug('-----------------------')


    print('Number of rows in full dataset: {} '.format(len(df)))
    print('Number of rows in train dataset: {} '.format(len(train)))
    print('Number of rows in test dataset: {} '.format(len(test)))
    print('Sum train + test : {} '.format(len(train) + len(test)))
    logging.debug('Number of rows in full dataset: {} '.format(len(df)))
    logging.debug('Number of rows in train dataset: {} '.format(len(train)))
    logging.debug('Number of rows in test dataset: {} '.format(len(test)))
    logging.debug('Sum train + test : {} '.format(len(train) + len(test)))

    return 0


def test_sub():

    str_origin = 'Allied success against Japan.\nAfter the Doolittle Raid, the Japanese army conducted'
    print('str_origin: {} '.format(str_origin))

    str_sub = str_origin.replace("\n", " ")
    print('str_sub: {} '.format(str_sub))