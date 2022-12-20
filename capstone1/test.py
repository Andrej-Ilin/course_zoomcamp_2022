import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# data = {'url': 'https://glorypets.ru/wp-content/uploads/2020/07/1-tsarstvennost.jpg'}
data = {'url': 'https://sun9-79.userapi.com/c11422/u1430261/148960630/x_b6d7669e.jpg'}
result = requests.post(url, json=data).json()
print(result)
#%%
