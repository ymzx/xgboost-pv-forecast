import pandas as pd
import numpy as np

need_col_name = ['ds', 'weather', 'humidity', 'pressure', 'realFeel', 'pop', 'temp', 'uvi', 'windDegrees', 'windSpeed', 'windLevel']

weather = {'中雪': 0, '雪': 1, '大雪': 2, '暴雨': 3,
           '雨': 4, '小雨': 5, '中雨': 6, '大雨': 7, '雨夹雪': 8, '阵雪': 9, '小雪': 10,
           '阴': 11, '霾': 12, '雾': 13, '阵雨': 14, '雷阵雨': 14, '浮尘': 15, '扬沙': 16,
           '多云': 17, '晴': 18}


def normalization(pd_data, features=[]):
    max_min_scaler = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    for name in features:
        pd_data[name] = pd_data[[name]].apply(max_min_scaler)
    return pd_data


def weather_mapfun(x):
    return weather[x]


def ds_mapfun(x, gap=15):
    scale = 60/gap
    idx = x.hour*scale + x.minute/gap
    return idx


def pv_mapfun(x):
    temp = x
    if x<=10: temp=0
    return temp


def read_pv_data(pv_excel_path, weather_excel_path, neighbour_points=8):
    pv_data = pd.read_excel(pv_excel_path).rename(columns={'y': 'pv'})
    weather_data = pd.read_excel(weather_excel_path).rename(columns={'sysTime': 'ds'})
    drop_names = []
    for col_name in list(weather_data.columns.values):
        if col_name not in need_col_name: # 选择需要的列作为特征
            drop_names.append(col_name)
        if col_name == 'weather':
            weather_data[col_name] = weather_data[col_name].map(weather_mapfun)
    weather_data.drop(drop_names, axis=1, inplace=True)
    # 数据关联
    merge_data = pd.merge(pv_data, weather_data, on='ds', how='inner')
    # 时间特殊处理,并将小于100置为0
    merge_data['date'] = merge_data['ds']
    merge_data['ds'] = merge_data['ds'].map(ds_mapfun)
    merge_data['pv'] = merge_data['pv'].map(pv_mapfun)
    # 特征归一化
    cols = list(merge_data.columns.values)
    without_normal_cols_name = ['pv', 'date']
    normal_cols_name = [ele for ele in cols if ele not in without_normal_cols_name]
    normal_merge_data = normalization(merge_data, normal_cols_name)
    if neighbour_points:
        for idx in range(neighbour_points):
            shift_target = normal_merge_data['pv'].shift(periods=idx + 1)
            normal_merge_data['dist' + '_' + str(idx + 1)] = shift_target
    normal_merge_data.dropna(axis=0, how='all', inplace=True)
    return normal_merge_data


if __name__ == '__main__':
    pv_excel_path = r'data/pv.xlsx'
    weather_excel_path = r'data/weather.xlsx'
    data = read_pv_data(pv_excel_path, weather_excel_path, neighbour_points=4)
    print(data.head(50))
