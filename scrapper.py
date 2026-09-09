import sys

import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NEWS_URL = "https://www.moneycontrol.com/news/business/markets"
MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_CSV = "moneycontrol_sentiment.csv"

# (connect, read) timeout in seconds - without this a hung server would block forever.
REQUEST_TIMEOUT = (5, 15)

# The markup we depend on. Kept here so failures can name exactly what stopped
# matching, and so a Moneycontrol layout change is a one-line fix. Documented in
# the "Scraping Target and Selectors" section of README.md.
ARTICLE_TAG = "li"
ARTICLE_CLASS = "clearfix"
ARTICLE_SELECTOR = f"{ARTICLE_TAG}.{ARTICLE_CLASS}"
TITLE_SELECTOR = "h2"


class NewsScrapeError(RuntimeError):
    """Raised when the news page cannot be fetched or yields no headlines."""


def get_moneycontrol_news(url=NEWS_URL):
    """Scrape the Moneycontrol markets page and return the headlines found."""
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise NewsScrapeError(f"Failed to fetch news from {url}: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all(ARTICLE_TAG, class_=ARTICLE_CLASS)
    headlines = []
    for article in articles:
        title_tag = article.find(TITLE_SELECTOR)
        if title_tag:
            headlines.append(title_tag.text.strip())

    if not headlines:
        raise NewsScrapeError(
            f"No headlines parsed from {url} (final URL: {response.url}, "
            f"HTTP {response.status_code}, {len(response.text)} bytes). "
            f"Matched {len(articles)} '{ARTICLE_SELECTOR}' elements and 0 usable "
            f"'{TITLE_SELECTOR}' titles inside them - the page layout probably "
            "changed, or the request was served a block/consent page."
        )

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
    try:
        news_headlines = get_moneycontrol_news()
    except NewsScrapeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

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
