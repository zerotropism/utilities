import requests


res = requests.get(
    "https://search-target1-zrhgle3vemuqzhhs7fudth7cae.eu-west-3.es.amazonaws.com/_cat/indices?v"
)
print(res.status_code)
if res.status_code != 200:
    print("ERROR during GET indices")
    exit(1)
else:
    print("element correctly GET indices!")

print(res.text)
