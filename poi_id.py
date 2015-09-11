#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score
from tester import dump_classifier_and_data
import math
import matplotlib.pyplot as plt
import numpy as np 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Remove outliers
# print keys and assess by eye for anything unusual
for name_key, value in data_dict.iteritems():
	print name_key

data_dict.pop('TOTAL')

### create new features - proporion of emails correspondence with poi 
my_dataset = data_dict

for key_name, values_features in my_dataset.iteritems():
	fraction_from_poi = float(values_features['from_poi_to_this_person']) / float(values_features['from_messages'])
	fraction_cc = float(values_features['shared_receipt_with_poi']) / float(values_features['from_messages'])
	fraction_to_poi = float(values_features['from_this_person_to_poi']) / float(values_features['to_messages'])

	values_features['fraction_from_poi'] = fraction_from_poi if math.isnan(fraction_from_poi) == False else 0
	values_features['fraction_cc'] = fraction_cc if math.isnan(fraction_cc) == False else 0
	values_features['fraction_to_poi'] = fraction_to_poi if math.isnan(fraction_to_poi) == False else 0

### select features using intuition, and then trial and error
features_list = ['poi','salary', 'bonus', 'exercised_stock_options', 'fraction_to_poi', 'fraction_from_poi', 'fraction_cc', 'expenses', 'loan_advances', 'other'] 


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


recall_list = []
precision_list = []
### use statified shuffle split to ensure all data used  as small sample 
### addiitonally, this will give a good spread of pois to test algorithm
sss = StratifiedShuffleSplit(labels, 10, test_size = 0.25, random_state = 0)
for train_index, test_index in sss:
	features_train = [features[i] for i in train_index]
	features_test = [features[i] for i in test_index]
	labels_train = [labels[i] for i in train_index]
	labels_test = [labels[i] for i in test_index]

	### use pca to reduce dimensions and noise
	pca = PCA(n_components = 3)
	### use classifier to make predicitons 
	random = RandomForestClassifier(n_estimators = 1500, min_samples_split = 3)
	clf = Pipeline([('pca', pca), ('clf', random)])
	features_train = clf.fit(features_train, labels_train).transform(features_train)
	pred = clf.predict(features_test)
	recall_list.append(recall_score(labels_test, pred))
	precision_list.append(precision_score(labels_test, pred))

### calculate average recall and precision over all folds
print np.mean(recall_list)
print np.mean(precision_list)

dump_classifier_and_data(clf, my_dataset, features_list)
