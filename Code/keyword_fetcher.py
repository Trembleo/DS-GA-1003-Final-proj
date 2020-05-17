import os
import numpy as np
import pandas as pd
import json
import nltk
import time
from tqdm import tqdm
from textblob import TextBlob
from pytrends.request import TrendReq

class KeywordFetcher():
    def __init__(self, term_path: str, bigram_path: str, trigram_path: str):
        self.term_path = term_path
        self.bigram_path = bigram_path
        self.trigram_path = trigram_path


    def generate_keyword_list(self, num_features: int, save_path=None)-> pd.Dataframe:
        """
        Output: 
            DataFrame of two columns ('keyword', 'count')
        """
        terms_data = pd.read_csv(self.term_path)
        terms = list(terms_data['term'])
        counts = list(terms_data['counts'])
        result1 = pd.DataFrame(np.array([terms, counts]).T, columns=['keyword', 'count'])[:num_features]

        bigram_data = pd.read_csv(self.bigram_path)
        bigrams = list(bigram_data['gram'])
        counts = list(bigram_data['counts'])
        result2 = pd.DataFrame(np.array([bigrams, counts]).T, columns=['keyword', 'count'])[:num_features]

        trigram_data = pd.read_csv(self.trigram_path)
        trigrams = list(trigram_data['gram'])
        counts = list(trigram_data['counts'])
        result3 = pd.DataFrame(np.array([trigrams, counts]).T, columns=['keyword', 'count'])[:num_features]

        keywords = pd.concat([result1, result2, result3], ignore_index=True)
        if save_path:
            keywords.to_csv(save_path)
        return keywords
    

    def fetch_trends(self, keyword_list: pd.DataFrame, start_date: str, end_date: str, region='', save_path=None, tz=300, **kwargs) -> pd.DataFrame:
        """
        Input:
            date format: 
                "YYYY-MM-DD"
            region: 
                Two letter country abbreviation
                e.g. 'US', 'US-NY'
        Output: 
            DataFrame, each row for one date, each column for one keyword.
        """
        ptr = TrendReq(hl='en-US', tz=tz, **kwargs) # tz: time-zone offset, 300 for EST
        kw_list = list(keyword_list["keyword"])[0]
        ptr.build_payload([kw_list], cat=0, timeframe="{} {}".format(start_date, end_date), geo='', gprop='')
        result = ptr.interest_over_time().drop(columns=['isPartial'])

        t = tqdm(list(keyword_list["keyword"])[1:])
        for kw in t:
            t.set_description(desc="Fetching...")
            del(ptr)
            time.sleep(0.5)
            ptr = TrendReq(hl='en-US', tz=300, retries=3, backoff_factor=10) # tz: time-zone offset, 300 for EST
            ptr.build_payload([kw], cat=0, timeframe='2020-01-21 2020-05-1', geo='', gprop='')
            try:
                result = pd.concat([result, ptr.interest_over_time().drop(columns='isPartial')], axis=1)
            except KeyError:
                result = pd.concat([result, ptr.interest_over_time()], axis=1)

        if save_path:
            result.to_csv(save_path)
        return result