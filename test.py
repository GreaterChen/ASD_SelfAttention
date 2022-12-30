import requests

url = "http://localhost:3000/user/account/token/"
d = {"username": "clb", "password": "clb030108"}
res = requests.post(url, d)
print(res.reason)
