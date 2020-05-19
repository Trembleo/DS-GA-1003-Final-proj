import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from datetime import date, timedelta

class DataParser:

    def __init__(self, keywords_path: str, cases_path: str, start_date: str, end_date:str,
                 avg_date=7, concat_date=7):
        """
        Input:
            date format: 
                "YYYY-MM-DD"
        """
        self.avg_date = avg_date
        self.concat_date = concat_date

        self.keywords = pd.read_csv(keywords_path)
        self.cases = pd.read_csv(cases_path)
        self.sdate = date.fromisoformat(start_date)
        self.edate = date.fromisoformat(end_date)


    # Average with date
    def process_data_sample(self, label_array, process='mean'):
        matrix = self.keywords.drop(['date'], axis=1,inplace=False).to_numpy()

        if process == 'mean':
            avg_mtx = np.zeros((matrix.shape[0]-self.avg_date, matrix.shape[1]))
            for i in range(len(label_array)):
                avg_mtx[i] = np.mean(matrix[i:i+self.avg_date],axis=0)
            # Standardrize mean-matrix
            # sample_matrix = scipy.stats.zscore(avg_mtx, axis=1)
            # sample_matrix = avg_mtx
            sample_matrix = avg_mtx / 100
        elif process == 'concat':
            concat_mtx =  np.zeros((matrix.shape[0]-self.concat_date,matrix.shape[1]*self.concat_date))
            for i in range(len(label_array)):
                concat_mtx[i] = np.ndarray.flatten(df_matrix[i:i+self.concat_date])
            sample_matrix = scipy.stats.zscore(concat_mtx, axis=1)
        return sample_matrix

    def process_labels(self, tag=None):
        if tag is None:
            tag = 'positiveIncrease'
        labels = self.cases[tag]
        avg_array_rev = np.zeros((labels.shape[0]-self.avg_date))
        for i in range(len(avg_array_rev)):
            avg_array_rev[i] = np.mean(labels[i:i+self.avg_date])
        label_array = np.flipud(avg_array_rev)[self.delay_date+1:]
        return label_array
    
    def process_labels_state(self, delay=0, delaytag='positiveIncrease'):
        y = []
        label = self.cases.fillna(0)
        for index, row in self.keywords.iterrows():
            state = row['state']
            iso_date = row['date']
            if date.fromisoformat(iso_date) < self.sdate or date.fromisoformat(iso_date) > self.edate:
                continue
            adate = date.fromisoformat(iso_date) + timedelta(days=delay)
            int_date = int(str(adate.year) + str(adate.month).zfill(2) + str(adate.day).zfill(2))
            label_row = label.loc[(label['state']==state) & (label['date']==int_date)]
            assert(len(label_row)==0 or len(label_row)==1)
            if label_row.empty:
                y.append(0)
            else:
                y.append(label_row['positiveIncrease'].values[0])
        return y
