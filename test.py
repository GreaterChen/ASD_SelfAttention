import requests
import urllib


def register_test():
    # url = "https://app2799.acapp.acwing.com.cn/api/user/account/register/"
    url = 'http://localhost:3000/api/user/account/register/'
    d = {"username": "tttt",
         "password": "clb030108",
         "confirmPassword": "clb030108",
         "mail": "1796390642@qq.com",
         "phone": "18056199338"}
    res = requests.post(url, d)
    print(res.reason)
    error_message = eval(res.text)['error_message']
    print(error_message)


def login_test():
    url = "https://app2799.acapp.acwing.com.cn/api/user/account/token/"
    d = {"username": "clb",
         "password": "clb030108"}

    res = requests.post(url, d)
    print(res.reason)
    res = res.text
    token = eval(res)['token']
    return token


def getinfo_test(token):
    url = "https://app2799.acapp.acwing.com.cn/api/user/account/info/"
    headers = {
        "Authorization": "Bearer " + token,
    }
    res = requests.get(url, headers=headers)
    value = eval(res.text)
    print(value)


if __name__ == '__main__':
    register_test()
    # getinfo_test(login_test())
