import requests
import os
import argparse
import json
import unidecode
import re
import csv
import pandas as pd
import string
import spacy


#getting user inputs 
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory where JSON are stored', type=str)
args = parser.parse_args()

es_index = 'discovery_dev_clean' # equivalent to SQL database
es_type = '_doc' # equivalent to SQL table: mandatory to use _doc for text elements
es_id_num_init = 1

# List all files in a directory using os.listdir
with os.scandir(args.data_dir) as files:
    for file in files:
        if file.is_file():
            with open(args.data_dir+'/'+file.name) as json_file:
                data = json.load(json_file)
                res = requests.post(
                    'https://search-target1-zrhgle3vemuqzhhs7fudth7cae.eu-west-3.es.amazonaws.com/'
                    + es_index +'/'
                    + es_type +'/'
                    + str(es_id_num_init),
                    json=data
                    )
                es_id_num_init = es_id_num_init + 1
    
                if((res.status_code == 200) or (res.status_code == 201) ):
                    print('File {} correctly added in index {}'.format(file.name, es_index))
                else:
                    print('ERROR during POSTING file {} error !'.format(file.name))
                    exit(1)