import os
import numpy as np
import pandas as pd
import json
import nltk
import time
from tqdm import tqdm
from textblob import TextBlob
from pytrends.request import TrendReq

class NgramFetcher():
    