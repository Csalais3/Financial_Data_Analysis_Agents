#My API key is: 3ea792e2a21c4a54a621191dd55a283d
import requests
import csv

# Sets API key
api_key = '3ea792e2a21c4a54a621191dd55a283d'

# Sets up the endpoint and parameters
url = 'https://newsapi.org/v2/top-headlines'
params = {
    'country': 'us',
    'category': 'business',  # Changes the country and topic, good to see what kind of results show up
    'apiKey': api_key
}

# Makes the request to NewsAPI
response = requests.get(url, params)

#  Response code 200 means successful connection
if response.status_code == 200:
    data = response.json()
    articles = data.get('articles', [])
    
    # Opens a CSV file to write the data
    with open('news_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['source', 'author', 'title', 'description', 'url', 'publishedAt', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Logs each article's relevant fields
        for article in articles:
            writer.writerow({
                'source': article.get('source', {}).get('name', ''),
                'author': article.get('author', ''),
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'content': article.get('content', '')
            })
    print("Data logged to news_data.csv successfully.")
else:
    print("Error:", response.status_code, response.text)

