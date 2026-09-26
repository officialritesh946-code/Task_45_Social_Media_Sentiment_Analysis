# Social Media Sentiment Analysis on a Trending Topic

## Project
**Airline Customer Experience & Service Conversations**

### Objective
Analyze social-media conversations using positive/neutral/negative sentiment labels; compare airlines, inspect sentiment over time, identify negative service issues, and extract common keywords.

### Dataset
- Source file supplied by the user: `Tweets.csv`
- Rows after duplicate removal: **14,485**
- Airlines: **6**
- Sentiment labels: positive, neutral, negative
- Data period: **February 2015**

### Scope note
The supplied dataset is historical. It is suitable for demonstrating the requested internship workflow, but it should not be described as current 2026 social-media sentiment or a current trending event. If the evaluator strictly requires a recent trend, the same pipeline should be rerun on a recent approved social-media dataset/API export.

### Key results
- Negative: **9,082 (62.70%)**
- Neutral: **3,069 (21.19%)**
- Positive: **2,334 (16.11%)**
- Highest negative-share airline in this dataset: **US Airways**
- Highest positive-share airline in this dataset: **Virgin America**
- Most common recorded negative reason: **Customer Service Issue**

### Deliverables
- `01_data/airline_sentiment_cleaned.csv`
- `02_analysis/social_media_sentiment_analysis.py`
- `02_analysis/Social_Media_Sentiment_Analysis.ipynb`
- `03_outputs/` charts and summary tables
- `04_report/Project_Report.pdf`
- `requirements.txt`

### Run
```bash
pip install -r requirements.txt
jupyter notebook 02_analysis/Social_Media_Sentiment_Analysis.ipynb
```
