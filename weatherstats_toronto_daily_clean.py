# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:41:52 2022

@author: Andres Lojan Yepez
"""
# Import necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

# Import dataset

dataset = pd.read_csv(r"C:\Users\Andres Lojan Yepez\Desktop\Assignments\Math concepts DL-I\Project\weatherstats_toronto_daily.csv")
dataset.info()

# Preprocessing and feature selection

# Sort data
dataset['date'] = pd.to_datetime(dataset['date'])
dataset = dataset.sort_values(by='date', ignore_index=True)

# Drop out hourly features
ds_daily = dataset.drop(['avg_hourly_temperature','avg_hourly_relative_humidity',
                         'avg_hourly_dew_point', 'avg_hourly_wind_speed',
                         'avg_hourly_pressure_sea', 'avg_hourly_pressure_station',
                         'avg_hourly_visibility', 'avg_hourly_health_index',
                         'avg_hourly_cloud_cover_4', 'avg_hourly_cloud_cover_8',
                         'avg_hourly_cloud_cover_10',], axis=1)

# Drop out single features
single_features = ['max_humidex', 'min_windchill', 'max_wind_gust', 'wind_gust_dir_10s',
                   'heatdegdays', 'cooldegdays', 'growdegdays_5', 'growdegdays_7',
                   'growdegdays_10', 'snow_on_ground', 'rain', 'snow', 'sunrise',
                   'sunset', 'daylight', 'sunrise_f', 'sunset_f', 'min_uv_forecast',
                   'max_uv_forecast', 'min_high_temperature_forecast', 'max_high_temperature_forecast',
                   'min_low_temperature_forecast', 'max_low_temperature_forecast',
                   'solar_radiation']
ds_daily_nosingleft = ds_daily.drop(single_features, axis=1)

# Drop out irrelevant features
irrelevant_features = ['max_pressure_station', 'avg_pressure_station', 'min_pressure_station',
                       'max_visibility', 'avg_visibility', 'min_visibility',
                       'max_health_index', 'avg_health_index', 'min_health_index']
ds_daily_nosingleft_relevant = ds_daily_nosingleft.drop(irrelevant_features, axis=1)

# Search for NaN values
ds_daily_nosingleft_relevant.isnull().sum()

# Drop out columns with no data (meaningfull amount of NaN)
ds_daily_nosingleft_relevant_nocloud = ds_daily_nosingleft_relevant.iloc[:, 0:17]

# Imputation for NaN values in 'precipitation'
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp = imp.fit(ds_daily_nosingleft_relevant_nocloud[['precipitation']])
ds_daily_nosingleft_relevant_nocloud['precipitation'] = imp.transform(ds_daily_nosingleft_relevant_nocloud[['precipitation']])
print(ds_daily_nosingleft_relevant_nocloud.isnull().sum())
print(ds_daily_nosingleft_relevant_nocloud.info())

# Export clean dataset
weatherstats_toronto_daily_clean = ds_daily_nosingleft_relevant_nocloud
weatherstats_toronto_daily_clean.to_csv('weatherstats_toronto_daily_clean.csv', index=False)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(weatherstats_toronto_daily_clean)
df_scaled = scaler.transform(weatherstats_toronto_daily_clean)