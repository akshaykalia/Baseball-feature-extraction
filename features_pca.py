import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import csv


print "Script start at ", datetime.now().isoformat()

df = pd.read_csv('training.csv')

names = ['BatAge','Batting_H','2B','3B','Batting_HR','SB','CS','Batting_BB','Batting_SO','BA','OBP','SLG','OPS','GDP','Batting_HBP','SH','SF','Batting_IBB','Batting_LOB','DefEff','Ch','A','E','DP','FldPercentage','Rtot','RtotPerYr','PAge','CG','tSho','cSho','IP','Pitching_H','Pitching_HR','Pitching_BB','Pitching_IBB','Pitching_SO','Pitching_HBP','BK','WP','ERA+','FIP','WHIP','H9','HR9','BB9','SO9','SOW','Pitching_LOB']
cols_to_keep = names
scores = []
df = df.dropna(axis=0, how='any')
X = df[names]
Y = df['WinPercentage']
ranks = {}

fout = open('out.txt','w')
#Convert the target to numpy array
arr_target = Y.as_matrix()

#Convert the dataframe to numpy array
arr_X = X.as_matrix()

arr_X_train, arr_X_val, arr_target_train, arr_target_val = train_test_split(arr_X, arr_target, test_size=0.1, random_state=20)

#PCA
num_comp = []
per_variance = []
fout.write('Variance vs No. Of Features\n')
fout.write('-'*30 + '\n')
for n in (0.99, 0.95, 0.90, 0.75, 0.65, 0.5):
	p=PCA(n_components=n).fit(arr_X_train)
	#print(p.explained_variance_)
	print n*100, len(p.explained_variance_)
	fout.write(str(n*100) + '\t' + str(len(p.explained_variance_)) + '\n')
	num_comp.append(len(p.explained_variance_))
	per_variance.append(n*100)

#num_comp.append(11) #the entire feature set
#hyperparameters
pca_val=num_comp
alpha_val = np.logspace(-5,5, num=11, base=2)
c_val = np.logspace(-5,5, num=11, base=2) #c for SVL and SVG
g_val = np.logspace(-5,5, num=11, base=2) #gamma for SVG

def runModel(model):
	if model == 'lr':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('lr',linear_model.LinearRegression())])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
	elif model == 'rr':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('rr',linear_model.Ridge())])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val,rr__alpha=alpha_val), cv=10)
	elif model == 'rf':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('rf', RandomForestRegressor(n_estimators=20, max_depth=4,random_state =5))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
	elif model == 'svrl':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('svr_lin',SVR(kernel='linear',C=1))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val,svr_lin__C=c_val), cv=10)
	elif model == 'svrg':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('svr_gaussian',SVR(kernel='rbf',C=1,gamma=1))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val,svr_gaussian__C=c_val,svr_gaussian__gamma=g_val), cv=10)
	elif model == 'adaboostRF':
		pipe=Pipeline([('pca',PCA()),('scaled',StandardScaler()), ('adaboost', AdaBoostRegressor(RandomForestRegressor(), random_state=0))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
	elif model == 'adaboostSVR':
		pipe=Pipeline([('pca',PCA()),('scaled',StandardScaler()), ('svr_adaboost',AdaBoostRegressor(SVR(), random_state=0))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
		
	gs.fit(arr_X_train, arr_target_train)

	#print gs.predict(arr_X_val)
	predictions = gs.predict(arr_X_val)
	print gs.score(arr_X_val,arr_target_val)
	print 'best_score'
	print gs.best_score_
	fout.write(str(gs.best_score_) + '\n')
	print 'best_estimator'
	print gs.best_estimator_
	print 'best_params'
	print gs.best_params_

fout.write('\n\nDifferent Model Accuracy\n')
fout.write('-'*30 + '\n')
models = ['lr','rr','rf','adaboostRF','adaboostSVR','svrl','svrg']
for model in models:
	print '*'*30
	print 'Running : ' , model
	print '*'*30 
	fout.write(model + '\t')
	runModel(model)

fout.write('\n\nImportant Features :\n')
fout.write('-'*30 + '\n')
#Find the k best features
k_val = list(set(num_comp))
for j in range(0,len(k_val)):
	#if k_val[j] != 9:
	contributingFeatures = []
	skb = SelectKBest(f_regression, k = k_val[j])
	arr_X_train_reshape = skb.fit(arr_X_train, arr_target_train)
	#arr_patrons_sales_events_val_reshape = skb.transform(arr_patrons_sales_events_val)
	print 'The top ', k_val[j], ' features are: '
	fout.write('The top ' + str(k_val[j]) + ' features are: \n')
	get_features = skb.get_support() #print True or False for the features depending on whether it matters for predicting the category or not
	for i in range(0,len(get_features)):
		if get_features[i]:
			contributingFeatures.append(cols_to_keep[i])
			print i, cols_to_keep[i]
			fout.write(cols_to_keep[i] + '\n')
	fout.write('\n')

