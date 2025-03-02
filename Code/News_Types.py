# Dependencies
import numpy as np
import pandas as pd
import requests
import csv
from collections import Counter
import re
from sklearn.neighbors import NearestNeighbors
import hdbscan


############################################################################################
#                             Implementing BERTopic With HDBSCAN
#-------------------------------------------------------------------------------------------
# - Goal: To Use Heirarchical Clustering to Create Most Relevant News Classifiers 
# 
# 
#
#                                 Supporting Documentation
#-------------------------------------------------------------------------------------------
#   https://papers.nips.cc/paper/2010/hash/b534ba68236ba543ae44b22bd110a1d6-Abstract.html
#       https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html 
#               https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf 
#               https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14 
#                           https://dl.acm.org/doi/10.1145/2733381 
#                               https://arxiv.org/abs/1911.02282
#                               https://arxiv.org/abs/2203.05794
#
#############################################################################################

# Variables For NEWSAPI:
api_key = '3ea792e2a21c4a54a621191dd55a283d'
url = 'https://newsapi.org/v2/top-headlines'
params = {
    'country': 'us',
    'category': 'business',  # Changes the country and topic, good to see what kind of results show up
    'apiKey': api_key
}

# Variables for HDBSCAN:
threshold_percentile = 75
min_cluster_size = 2
min_samples = 10

# Other
DATAFILE = "/Users/csalais3/Downloads/Financial_Data_Analysis_Agents/Data/US500_Historical_Data.csv"

class NewsClassifier:
    def __init__(self, df):
        df = self.df
    
    def preprocessing(self, text):
        text = text.lower()
        text = re.sub(r'[a-z\s]', '', text)
        words = text.split()
        
        return words
    
    def build_vocab(self, processedDocs):
        vocab = {}
        
        for doc in processedDocs:
            for word in doc:
                if word not in vocab:
                    vocab[word] = len(word)

        return vocab
    
    def calculate_term_frequency(self, doc, vocab, vocab_size):
        counts = Counter(doc)
        tf_vector = np.zeros(vocab_size)
        
        for word, count in counts.items():
            idx = vocab[word]
            tf_vector[idx] = count
        
        return tf_vector
    
    def calculate_IDF(self, processedDocs, tf_matrix):
        N = len(processedDocs)
        df_counts = np.sum(tf_matrix > 0, axis = 0)
        idf = np.log((N + 1) / (df_counts + 1)) + 1
        
        return idf
    
    def hdbscan(self, data):
        pass
    
    def get_top_keywords(self, clustertfidf, inv_vocab, top_n= 10):
        summed_tfidf = np.sum(clustertfidf, axis = 0)
        top_words_idx = np.argsort(summed_tfidf)[::-1][:top_n]
        
        return [inv_vocab[idx] for idx in top_words_idx]
    
    def run_classifier(self, df):
        documents = (df["title"].fillna() + "" + df["description"].fillna()).tolist()
        processedDocs = [self.preprocessing(doc) for doc in documents]
        
        vocab = self.build_vocab(processedDocs)
        inv_vocab = {idx: word for word, idx in vocab.items()}
        vocab_size = len(vocab)
        print("Our Vocab Size: ", vocab_size)
        
        tfMatrix = [self.calculate_term_frequency(doc, vocab, vocab_size) for doc in processedDocs]
        tfidfMatrix = tfMatrix * self.calculate_IDF(processedDocs, tfMatrix)
        norm = np.linalg.norm(tfidfMatrix, axis= 1, keepdims= True)
        norm_tfidfMatrix = tfidfMatrix / (norm + 1e-10)
        
        clusters = self.hdbscan(norm_tfidfMatrix)
        clusterd_keywords = {}
        for cluster in range(len(clusters)):
            cluster_docs = tfidfMatrix[df['cluster'] == cluster]
            if cluster_docs.shape[0] > 0:
                keywords = self.get_top_keywords(cluster_docs)
                clusterd_keywords[cluster] = keywords
                print(f"Cluster {cluster}: {','.join(keywords)}")
        
        df.to_csv("clustered_news.csv")
        

def news_request(params):
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


df = pd.read_csv(DATAFILE)
