import requests
import argparse
import os


# getting user inputs
parser = argparse.ArgumentParser()
parser.add_argument("--index", help='Elasticsearch index name"', type=str)
parser.add_argument("--type", help='Elasticsearch type"', type=str)
args = parser.parse_args()

es_index = args.index
es_type = args.type

res = requests.delete(
    os.environ["ES_HOST"]
    + es_index
)

print(res.status_code)
if res.status_code != 200:
    print("ERROR during DELETE index")
    exit(1)
else:
    print("element correctly DELETED index!")
