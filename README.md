# Predicting Price Moves with News Sentiment

## Overview
This project explores how financial news headlines impact stock price movements. By combining **Natural Language Processing (NLP)** techniques, **technical analysis indicators**, and **correlation studies**, this project aims to quantify news sentiment and link it to stock returns to provide actionable insights for trading and investment strategies.

The challenge refines skills in:
- **Data Engineering (DE)**
- **Financial Analytics (FA)**
- **Machine Learning Engineering (MLE)**

---

## Business Objective
Nova Financial Solutions aims to enhance predictive analytics for improved forecasting accuracy and operational efficiency. As a data analyst, the project goal is to:
1. Perform sentiment analysis on financial news headlines.
2. Establish correlations between sentiment and stock price movements.
3. Recommend strategies based on insights from news sentiment trends.

---

## Dataset
**Financial News and Stock Price Integration Dataset (FNSPID)**  
The dataset combines financial news headlines with stock price data.  

**Columns:**
- `headline` – Title of the news article
- `url` – Link to the full article
- `publisher` – Author or organization
- `date` – Publication date and time (UTC-4)
- `stock` – Stock ticker symbol (e.g., AAPL)


---

## Week-1 Tasks

### **Task 1: Exploratory Data Analysis (EDA)**
- Compute descriptive statistics (headline lengths, article counts per publisher).
- Analyze publication dates and times for trends.
- Conduct text analysis (topic modeling, keyword extraction).
- Identify active publishers and publisher domains.

**Key Libraries:** `pandas`, `matplotlib`, `seaborn`, `sklearn`, `nltk`

---

### **Task 2: Quantitative Analysis Using TA-Lib and PyNance**
- Load and prepare stock price data (`Open`, `High`, `Low`, `Close`, `Volume`).
- Calculate technical indicators:
  - Simple Moving Averages (SMA20, SMA50)
  - Relative Strength Index (RSI)
  - MACD and MACD histogram
- Compute additional financial metrics using PyNance (daily returns, volatility, moving averages).
- Visualize stock prices with indicators for trend analysis.

**Key Libraries:** `yfinance`, `talib`, `pynance`, `matplotlib`, `seaborn`

---

### **Task 3: Correlation Between News Sentiment & Stock Movement**
- Align news and stock price datasets by date.
- Perform sentiment analysis on headlines using NLTK:
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])

