import requests

url = "https://raw.githubusercontent.com/reinaldoq/processing-heart-disease-dataset/master/processed.cleveland.data"
headers = {'User-Agent': 'Mozilla/5.0'}
try:
    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code == 200:
        with open('heart.csv', 'wb') as f:
            f.write(response.content)
        print("Download successful")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
except Exception as e:
    print(f"An error occurred: {e}")
