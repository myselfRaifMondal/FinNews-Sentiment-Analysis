# FinNews Sentiment Analysis
## Overview
This repository contains a Python-based sentiment analysis tool that fetches financial news from Moneycontrol and determines the sentiment (positive, negative, or neutral) of the news articles. The sentiment analysis helps traders and investors make informed decisions based on the latest financial news.

# Features
- Scrapes financial news from Moneycontrol.
- Performs sentiment analysis using Natural Language Processing (NLP)
- Provides sentiment scores for each news article
- Supports visuallization of sentiment trends over time.
- Can be integrated with trading strategies for automated decision-making.

# Tech Stack
- Python: Core programming language.
- BeautifulSoup: Web scraping to fetch news articles.
- Requests: HTTP requests for fetching web data.
- FinBERT: Sentiment Analysis
- Pandas & Matplotlib: Data processing and visualization.

# Installation
1. Clone the repository
```
git clone https://github.com/myselfRaifMondal/FinNews-Sentiment-Analysis.git
cd FinNews-Sentiment-Analysis
```
2. Install dependecies
```
pip install -r requirements.txt
```

# Usage
1. Run the scrapper to run everything.
```
python scrapper.py
```

# Scraping Target and Selectors
The scraper depends on Moneycontrol's current HTML structure. If Moneycontrol changes
its markup, `scrapper.py` will find no headlines and exit with a `NewsScrapeError`
instead of producing results. Everything you need to repair it is listed here.

| What | Value | Defined in `scrapper.py` |
| --- | --- | --- |
| Page fetched | `https://www.moneycontrol.com/news/business/markets` | `NEWS_URL` |
| Article container | `li` elements with class `clearfix` (`li.clearfix`) | `ARTICLE_TAG` + `ARTICLE_CLASS` |
| Headline inside each container | the first `h2` element; its text is the headline | `TITLE_SELECTOR` |

Roughly, the parsing step is:

```python
soup = BeautifulSoup(response.text, "html.parser")
articles = soup.find_all(ARTICLE_TAG, class_=ARTICLE_CLASS)  # "li", "clearfix"
for article in articles:
    title_tag = article.find(TITLE_SELECTOR)                 # "h2"
    headlines.append(title_tag.text.strip())
```

## When it breaks
`get_moneycontrol_news()` raises `NewsScrapeError` when zero headlines are parsed. The
message reports the final URL, HTTP status, response size, and how many
`li.clearfix` elements matched, which distinguishes the two common failures:

- **0 `li.clearfix` matched** - the article container class changed. Open the page,
  find the element that now wraps each news item, and update `ARTICLE_TAG` /
  `ARTICLE_CLASS`.
- **Containers matched but no titles** - the headline tag changed. Update
  `TITLE_SELECTOR` to whatever tag now holds the headline text.
- **A small response or a redirect to another URL** - the request was served a
  block/consent page rather than the news listing; the `User-Agent` header in
  `get_moneycontrol_news()` may need updating.

Both selectors are module-level constants at the top of `scrapper.py` so a fix is a
one-line change.

# Future Enhancements
- Use machine learning for more advanced sentiment analysis
- Integrate with trading bots for automated trading decisions.
- Fetch data from multiple financial news sources.
- Deploy as a web application with real-time updates.

# Contributing
Feel free to contribute by submitting issues, feature requests, or pull requests.

# License
This project is licensed under the MIT License.

---
# Contact
For any queries or collaboration, reach out to ```raifmondal@icloud.com```