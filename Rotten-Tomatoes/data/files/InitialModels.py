#### imports
import pickle as pkl
import time, nltk, unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, cross_validate, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

### cleaning dataset for gradient boost, randomForest, and LogisticRegression

df = pd.read_csv('rotten_tomatoes_reviews.csv')
df = df.drop_duplicates(subset=['Review', 'Freshness'])
y, X = df['Freshness'].to_numpy(), df["Review"]
X_train, X_test, y_train, y_test = train_test_split(clean(), test_size=0.2, shuffle=True)


def modeling(X_train, y_train):
    tfidf = TfidfVectorizer(analyzer='word',max_features=50000, stop_words='english')
    document_tfidf_matrix = tfidf.fit_transform(X_train)
    forest_params = {'n_estimators': 1800,
                    'min_samples_split': 2,
                    'min_samples_leaf': 2,
                    'cv' : 3,
                    'verbose' : 4
                    'max_features': 'auto',
                    'max_depth': None,
                    'n_jobs' : -1,
                    'bootstrap': True}

    gistic_params = {'max_iter' : 325,
                    'solve' : 'lbfgs',
                    'verbose' : 1,
                    'cv' : 3,
                    'scoring' : 'accuracy'}
    boost_params = {}

    models = [LogisticRegression(),
                RandomForestClassifier(),
                GradientBoostingClassifier()]

    for model in models:

        if model == LogisticRegression():
            log_reg_score = cross_validate(model, document_tfidf_matrix, y_train, fit_params=gistic_params)
            print('Logistic Regression Modeling Scores \n')
            for key, val in log_reg_score.items():
                print(f'The {} is: {val.mean()}')

        if model == RandomForestClassifier():
            random_forest_score = cross_validate(model, document_tfidf_matrix, y_train, fit_params=forest_params, scoring=['accuracy', 'precision', 'recall'])
            print('Random Forest Modeling Scores \n')
            for key, val in random_forest_score.items():
                print(f'The {} is: {val.mean()}')

        if model == GradientBoostingClassifier():
            gradient_boost_score = cross_validate(model, document_tfidf_matrix, y_train, fit_params=forest_params, scoring=['accuracy', 'precision', 'recall'])
            print('Gradient Boost Modeling Scores \n')
            for key, val in gradient_boost_score.items():
                print(f'The {} is: {val.mean()}')

