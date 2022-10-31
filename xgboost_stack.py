import pickle
from data_loader import read_pv_data
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np


class XgboostStack:
    def __init__(self, start_time, end_time, max_depth=3, n_estimators=1000, neighbour=8):
        self.start, self.end = start_time, end_time
        self.neighbour = neighbour # 预测
        self.pv_excel = r'data/pv.xlsx'
        self.weather_excel = r'data/weather.xlsx'
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.data = read_pv_data(pv_excel_path=self.pv_excel, weather_excel_path=self.weather_excel, neighbour_points=neighbour)
        model_name = self.start.replace('-', ' ').replace(':', ' ').replace(' ', '_') + '_' + str(max_depth) + '_' + str(n_estimators) + '.pkl'
        model_dir = 'model_files'
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        self.save_path = os.path.join(model_dir, model_name)

    def build_train_data(self):
        train_data = self.data[self.data['date'] <= pd.to_datetime(self.start)]
        labels = train_data['pv']
        features = train_data.drop(['pv', 'date'], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.001, random_state=42)
        return self.x_train, self.y_train

    def train(self):
        print('---------train------------')
        model = xgb.XGBRegressor(max_depth=self.max_depth, n_estimators=self.n_estimators, learning_rate=0.1)
        model.fit(self.x_train, self.y_train)
        model.save_model(self.save_path)
        pickle.dump(model, open(self.save_path, "wb"))

    def load_model(self):
        print('---------load model------------')
        self.infer_model = pickle.load(open(self.save_path, "rb"))

    def infer(self):
        print('---------infer------------')
        if self.end is None: self.end = pd.to_datetime(self.start) + datetime.timedelta(hours=4)
        pred_raw_data = self.data[(self.data['date'] > pd.to_datetime(self.start)) & (self.data['date'] <= pd.to_datetime(self.end))]
        pred_data = pred_raw_data.drop(['pv', 'date'], axis=1)
        self.date = pred_raw_data['date']
        self.gt = pred_raw_data['pv']
        nrow, ncol = pred_data.shape
        temp_feat = pred_data[0:1].iloc[0, -self.neighbour:].copy()
        self.pred = []
        if self.neighbour:
            for i in range(nrow):
                feature = pred_data[i:i+1].copy()
                feature.iloc[0, -self.neighbour:] = temp_feat
                output = self.infer_model.predict(feature) # numpy.ndarray
                temp_feat = temp_feat.shift(1)  # 移动一位
                temp_feat[0] = output[0]
                self.pred.append(output[0])
        else:
            feature = pred_data
            output = self.infer_model.predict(feature)
            self.pred += output.tolist()

    def metric(self):
        mse, mape = mean_squared_error(self.pred, self.gt), mean_absolute_percentage_error(self.pred, self.gt)
        wmape, diff = self.weighted_mean_absolute_pct_error(self.gt, self.pred, threshold=200)
        print('metric', round(mse, 3), round(1-mape, 3), round(1-wmape, 3))

    def plot(self):
        plt.plot(self.date, self.gt, label='gt')
        plt.plot(self.date, self.pred, label='pred')
        plt.legend()
        plt.show()

    def weighted_mean_absolute_pct_error(self, y_true, y_pred, threshold=0.0):
        """ 加权绝对百分比误差，实际值与预测值差值的绝对值除以序列所有实际值的平均值 """
        y_true = y_true.tolist()
        gt, pred, delta = [], [], []
        for i, ele in enumerate(y_true):
            if ele > threshold:
                gt.append(ele)
                pred.append(y_pred[i])
                delta.append(ele - y_pred[i])
        diff = np.abs(delta)
        tm = np.nanmean(np.abs(gt))
        if tm == 0: return 0, 0
        diff = diff / tm
        return np.nanmean(diff), diff


'''
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
'''

if __name__ == '__main__':
    train = True
    start_time = '2022-09-5 00:00:00'  # 预测起始时间
    end_time = '2022-09-15 23:45:00'  # 预测起始时间
    # end_time = None # 预测截至时间, 默认4h
    xgb_stack = XgboostStack(start_time=start_time, end_time=end_time, neighbour=4)
    xgb_stack.build_train_data()
    if train: xgb_stack.train()
    xgb_stack.load_model()
    xgb_stack.infer()
    xgb_stack.metric()
    xgb_stack.plot()
