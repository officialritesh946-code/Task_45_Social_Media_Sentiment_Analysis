"""
Social Media Sentiment Analysis on a Trending Topic
Airline Customer Experience & Service Conversations
"""

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

DATA = "01_data/airline_sentiment_cleaned.csv"
df = pd.read_csv(DATA)

# Download VADER lexicon once if it is not already installed.
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

# 1. Overall sentiment
order = ["positive", "neutral", "negative"]
df["airline_sentiment"].value_counts().reindex(order).plot(kind="bar")
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

# 2. Sentiment by airline
cross = pd.crosstab(df["airline"], df["airline_sentiment"]).reindex(columns=order, fill_value=0)
cross.plot(kind="bar", stacked=True)
plt.title("Sentiment by Airline")
plt.xlabel("Airline")
plt.ylabel("Tweet Count")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

# 3. Sentiment over time
df["date"] = pd.to_datetime(df["date"])
daily = pd.crosstab(df["date"], df["airline_sentiment"]).reindex(columns=order, fill_value=0)
daily_pct = daily.div(daily.sum(axis=1), axis=0) * 100
daily_pct.plot()
plt.title("Sentiment Share Over Time")
plt.xlabel("Date")
plt.ylabel("Share (%)")
plt.tight_layout()
plt.show()

# 4. VADER NLP validation
df["vader_compound"] = df["text"].fillna("").map(lambda x: sia.polarity_scores(x)["compound"])
df["vader_sentiment"] = df["vader_compound"].apply(
    lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
)
agreement = (df["vader_sentiment"] == df["airline_sentiment"]).mean() * 100
print(f"VADER agreement with provided labels: {agreement:.2f}%")

# 5. Negative reasons
reasons = df.loc[df["airline_sentiment"].eq("negative"), "negativereason"].value_counts().head(10)
reasons.sort_values().plot(kind="barh")
plt.title("Top Negative Reasons")
plt.xlabel("Tweet Count")
plt.tight_layout()
plt.show()

# 6. Negative keyword word cloud
negative_text = " ".join(df.loc[df["airline_sentiment"].eq("negative"), "text_clean"].astype(str))
wc = WordCloud(width=1400, height=800, background_color="white", collocations=False).generate(negative_text)
plt.figure(figsize=(14, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Tweet Keyword Word Cloud")
plt.show()
