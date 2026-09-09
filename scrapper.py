import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NEWS_URL = "https://www.moneycontrol.com/news/business/markets"
MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_CSV = "moneycontrol_sentiment.csv"


def get_moneycontrol_news(url=NEWS_URL):
    """Scrape the Moneycontrol markets page and return the headlines found."""
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for article in soup.find_all("li", class_="clearfix"):
        title_tag = article.find("h2")
        if title_tag:
            headlines.append(title_tag.text.strip())

    return headlines


def load_sentiment_pipeline(model_name=MODEL_NAME):
    """Load the FinBERT tone model and return a sentiment-analysis pipeline."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def analyze_headlines(headlines, nlp_pipeline=None):
    """Return the predicted sentiment label for each headline."""
    if nlp_pipeline is None:
        nlp_pipeline = load_sentiment_pipeline()

    return [nlp_pipeline(headline)[0]["label"] for headline in headlines]


def build_dataframe(headlines, sentiments):
    """Combine headlines and their sentiment labels into a DataFrame."""
    df = pd.DataFrame(headlines, columns=["Headline"])
    df["Sentiment"] = sentiments

    return df


def save_dataframe(df, path=OUTPUT_CSV):
    """Write the results to CSV."""
    df.to_csv(path, index=False)


def plot_sentiment_distribution(df):
    """Show a bar chart of the sentiment distribution (blocks until closed)."""
    sns.countplot(x=df["Sentiment"])
    plt.title("Sentiment Distribution of Moneycontrol News")
    plt.show()


def main():
    news_headlines = get_moneycontrol_news()
    for idx, headline in enumerate(news_headlines, 1):
        print(f"{idx}. {headline}")

    sentiments = analyze_headlines(news_headlines)

    for headline, sentiment in zip(news_headlines, sentiments):
        print(f"News: {headline} \nSentiment: {sentiment}\n")

    df = build_dataframe(news_headlines, sentiments)
    save_dataframe(df)
    print("Sentiment analysis saved to the file.")

    plot_sentiment_distribution(df)

    return df


if __name__ == "__main__":
    main()
