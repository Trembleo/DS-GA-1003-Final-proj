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

    def __init__(self, keywords_path: str, cases_path: str, start_date: str, end_date:str):
        """
        Input:
            date format: 
                "YYYY-MM-DD"
        """
        self.keywords = pd.read_csv(keywords_path)
        self.cases = pd.read_csv(cases_path)
        self.sdate = date.fromisoformat(start_date)
        self.edate = date.fromisoformat(end_date)

    # Average with date
    def process_data_sample(self, label_array, process=None, avg_date=0, concat_date=0):
        matrix = self.keywords.drop(['date'], axis=1,inplace=False).to_numpy()
        if process is None:
            sample_matrix = matrix[:len(label_array)] / 100
        elif process == 'mean':
            avg_mtx = np.zeros((len(label_array), matrix.shape[1]))
            for i in range(len(label_array)):
                avg_mtx[i] = np.mean(matrix[i:i + avg_date],axis=0)
            sample_matrix = avg_mtx / 100
        elif process == 'concat':
            concat_mtx =  np.zeros((len(label_array), matrix.shape[1] * concat_date))
            for i in range(len(label_array)):
                concat_mtx[i] = np.ndarray.flatten(matrix[i:i + concat_date])
            sample_matrix = avg_mtx / 100
        return sample_matrix


    def process_labels(self, delay=0, avg_date=None, tag=None):
        if tag is None:
            tag = 'positiveIncrease'
        labels = self.cases[tag]
        if avg_date is None:
            label_array = np.flipud(labels)[delay+1:]
        else:
            avg_array_rev = np.zeros((labels.shape[0] - avg_date))
            for i in range(len(avg_array_rev)):
                avg_array_rev[i] = np.mean(labels[i:i + avg_date])
            label_array = np.flipud(avg_array_rev)[delay+1:]
        return label_array


    def process_samples_state(self, normalize=True, include_population=True, add_yesterday=False):
        df = self.keywords.loc[(self.keywords['date'] >= self.sdate.isoformat()) & 
                                (self.keywords['date'] <= self.edate.isoformat())]
        if not include_population:
            df = df.drop(['population'], axis=1, inplace=False)
        df_mtx = df.drop(['date', 'Unnamed: 0', 'state'], axis=1, inplace=False).to_numpy()

        if add_yesterday:
            df_yesterday = df_mtx[1:, :]
            df_yesterday = np.vstack((np.zeros((1, df_mtx.shape[1])), df_mtx[1:, :]))
            df_mtx = np.hstack((df_mtx, df_yesterday))

        if normalize:
            return df_mtx / np.max(df_mtx, axis=0)
        return df_mtx


    def process_labels_state(self, delay=0, delaytag='positiveIncrease', normalize=False, divide_population=False):
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
                y_value = label_row['positiveIncrease'].values[0]
                if  divide_population:
                    y_value = y_value * 1e6 / row['population'] # number per million people
                y.append(y_value)

        if normalize:
            return np.array(y) / np.max(y)
        return np.array(y)
