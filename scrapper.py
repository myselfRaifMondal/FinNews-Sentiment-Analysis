import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_moneycontrol_news():
    url = "https://www.moneycontrol.com/news/business/markets"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for article in soup.find_all("li", class_="clearfix"):
        title_tag = article.find("h2")
        if title_tag:
            headlines.append(title_tag.text.strip())
    
    return headlines

news_headlines = get_moneycontrol_news()
for idx, headlines in enumerate(news_headlines, 1):
    print(f"{idx}. {headlines}")

model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

for headline in news_headlines:
    result = nlp_pipeline(headline)
    sentiment = result[0]['label']
    print(f"News: {headline} \nSentiment: {sentiment}\n")

df = pd.DataFrame(news_headlines, columns=["Headline"])
df["Sentiment"] = [nlp_pipeline(headline)[0]['label'] for headline in news_headlines]

df.to_csv("moneycontrol_sentiment.csv", index=False)
print("Sentiment analysis saved to the file.")

sns.countplot(x=df["Sentiment"])
plt.title("Sentiment Distribution of Moneycontrol News")
plt.show()