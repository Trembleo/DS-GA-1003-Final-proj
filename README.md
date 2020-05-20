# COVIDtrends: Increase patient predictor with internet searchtrend
## DS-GA 1003 Final Project
#### You Li (yl6911)
#### Arthur Jinyue Guo (jg5505)

## Repo Structure
- code
    - keyword_fetcher.py:
        contains class `KeywordFetcher`, which can generate keyword list from the frequent terms, and fetch corresponding Google Trends data.
    - data_parset.py:
        contains class `DataParser`, which can parse Google Trends data as X, and covid-19 cases data as y.
    - COVIDtrends.ipynb:
        contains the process of keyword fetching, data parsing, and model training. Some result plots are also contained.
    - other files:
        legacy code.
- data
    - frequent_terms.csv, frequent_bigrams.csv, frequent_trigrams.csv:
        The list of top 1000 n-grams from the COVID-19 twitter dataset.
    - covid19-in-usa:
        contains the cases data of US and every state.
    - trends:
        The data fetched from Google Trends. 30 and 150 means the number of features.

Resources:
 - Covid-19 Twitter dataset: https://github.com/thepanacealab/covid19_twitter 
 - Google Trends: https://trends.google.com/trends/?geo=US
 - Novel Coronavirus (COVID-19) Cases Data: https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
