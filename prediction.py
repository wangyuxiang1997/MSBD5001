# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:41:55 2019

@author: April
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score   #cross validation
from sklearn.model_selection import GridSearchCV 

# Importing the dataset
data_raw = pd.read_csv('E:/BDT/5001/Kaggle/raw_data/train.csv')
data_all = pd.read_csv('E:/BDT/5001/Kaggle/Data.csv')

X_train = data_all.iloc[0:357, 0:].values
y_train = data_raw.iloc[:, 1].values
X_test = data_all.iloc[357:,0:].values


"""X = data_all.iloc[0:357,1:].values
y = data_raw.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
#=====================================PCA===============================================
from sklearn.decomposition import PCA
pca = PCA(n_components = 11) # n_components = None
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# ==============================Linear============================================================
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)
accuracies = cross_val_score(estimator = linear, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_linear = np.sqrt(-accuracies).mean()

pred_linear = linear.predict(X_test)
df = pd.read_csv("E:/BDT/5001/Kaggle/samplesubmission.csv")
df['playtime_forever'] = list(pred_linear)
df.playtime_forever[df["playtime_forever"]<0] = 0
df.to_csv("E:/BDT/5001/Kaggle/submission3.csv", index = False)

#================================= LASSO===========================================================
"""from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.21, max_iter = 3)
lasso.fit(X_train, y_train)
accuracies = cross_val_score(estimator = lasso, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_lasso = np.sqrt(-accuracies).mean()

parameters = [{'alpha':[0.21,0.22,0.25,0.27,0.28],'max_iter':[10,9,8,7,6,5,4,3,2]}]
grid_search = GridSearchCV(estimator=lasso, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_


pred = lasso.predict(X_test)
df = pd.read_csv("E:/BDT/5001/Kaggle/samplesubmission.csv")
df['playtime_forever'] = list(pred)
df.playtime_forever[df["playtime_forever"]<0] = 0
df.to_csv("E:/BDT/5001/Kaggle/submission5.csv", index = False)"""

#===================================Ridge===========================================================
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.11,max_iter = 2)
ridge.fit(X_train, y_train)
accuracies = cross_val_score(estimator = ridge, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_ridge = np.sqrt(-accuracies).mean()

parameters = [{'alpha':[0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2],'max_iter':[2,3,4,5,6,7,8,9,10]}]
grid_search = GridSearchCV(estimator=lasso, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_

pred_ridge = ridge.predict(X_test)
df = pd.read_csv("E:/BDT/5001/Kaggle/samplesubmission.csv")
df['playtime_forever'] = list(pred_ridge)
df.playtime_forever[df["playtime_forever"]<0] = 0
df.to_csv("E:/BDT/5001/Kaggle/submission23.csv", index = False)

#==================================SVM===========================================================
"""from sklearn.svm import SVR
svr = SVR(C = 100,gamma = 0.08, kernel='rbf')
svr.fit(X_train, y_train)
accuracies = cross_val_score(estimator = svr, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_svr = np.sqrt(-accuracies).mean()

pred_svr = svr.predict(X_test)
df = pd.read_csv("E:/BDT/5001/Kaggle/samplesubmission.csv")
df['playtime_forever'] = list(pred_svr)
df.playtime_forever[df["playtime_forever"]<0] = 0
df.to_csv("E:/BDT/5001/Kaggle/submission11.csv", index = False)

parameters = [{'C':[80,90,100,110,120],'gamma':[0.08,0.09,0.1,0.11,0.12]}]
grid_search = GridSearchCV(estimator=svr, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_

#==================================Bayes========================================================
from sklearn.linear_model import BayesianRidge
bys = BayesianRidge()
bys.fit(X_train, y_train)
accuracies = cross_val_score(estimator = bys, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_bys = np.sqrt(-accuracies).mean()

#=================================Random forest==================================================
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=8, random_state=0)
rf.fit(X_train, y_train)
accuracies = cross_val_score(estimator = rf, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_rf = np.sqrt(-accuracies).mean()

#===============================XGB===================================================================
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=10, learning_rate=0.01)
xgb.fit(X_train, y_train)
accuracies = cross_val_score(estimator = xgb, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_xgb = np.sqrt(-accuracies).mean()

#=======================================GBM===============================================================
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=59, learning_rate=0.1,max_depth=3)
gbm.fit(X_train, y_train)
accuracies = cross_val_score(estimator = gbm, X=X_train, y=y_train, cv=10, scoring='neg_mean_squared_error')
rmse_gbm = np.sqrt(-accuracies).mean()

parameters = [{'n_estimators':range(20,50),'max_depth':range(2,10)}]
grid_search = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error',\
                           cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameter = grid_search.best_params_

pred_gbm = gbm.predict(X_test)
df = pd.read_csv("E:/BDT/5001/Kaggle/samplesubmission.csv")
df['playtime_forever'] = list(pred_gbm)
df.playtime_forever[df["playtime_forever"]<0] = 0
df.to_csv("E:/BDT/5001/Kaggle/submission16.csv", index = False)"""
