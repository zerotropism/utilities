import requests


res = requests.get('https://ES_HOST_PLACEHOLDER/_cat/indices?v')
print(res.status_code)
if res.status_code != 200:
    print('ERROR during GET indices')
    exit(1)
else:
    print('element correctly GET indices!')

print(res.text)