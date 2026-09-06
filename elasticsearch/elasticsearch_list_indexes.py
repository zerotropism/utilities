import requests
import os


res = requests.get(
    os.environ["ES_HOST"]
)
print(res.status_code)
if res.status_code != 200:
    print("ERROR during GET indices")
    exit(1)
else:
    print("element correctly GET indices!")

print(res.text)
