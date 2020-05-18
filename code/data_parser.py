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

    def __init__(self, keywords_path: str, cases_path: str, start_date: str,
                 avg_date=7, concat_date=7, delay_date=7):
        """
        Input:
            date format: 
                "YYYY-MM-DD"
        """
        self.avg_date = avg_date
        self.concat_date = concat_date
        self.delay_date = delay_date

        ## generate date list
        date1 = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]))
        # date2 = date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8]))
        # date_list = [str(date1 + timedelta(days=x)) for x in range((date2-date1).days + 1)]
        self.keywords = pd.read_csv(keywords_path)
        # self.keywords = self.keywords.loc[~self.keywords["date"].isin(date_list)]
        # date_list = [str(date1 + timedelta(days=x)) for x in range((date2-date1).days + 1 + delay_date)]
        self.cases = pd.read_csv(cases_path)
        # self.cases = self.cases.loc[~self.cases["date"].isin(date_list)]

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
