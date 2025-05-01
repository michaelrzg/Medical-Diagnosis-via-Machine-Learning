import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd

class BayesClassifier:
    def __init__(self):
        self.bayes = GaussianNB()

    def load_data(self):
        # Load the dataset
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = cancer.target
        # grab features

        features = list(X.columns)
        # Split the data into train, test, adn validation sets
        # lines 19 and 20 borrowed from sklearn website
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        xtrain, xvalidation, ytrain, yvalidation = train_test_split(xtrain, ytrain, test_size=0.25, random_state=42) # 0.25 of 0.8 is 0.2

        # normalize the data
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xvalidation = scaler.transform(xvalidation)
        xtest = scaler.transform(xtest)

        # Convert data to tensors
        self.xtrain = torch.tensor(xtrain, dtype=torch.float32)
        self.xvalidation = torch.tensor(xvalidation, dtype=torch.float32)
        self.xtest = torch.tensor(xtest, dtype=torch.float32)
        self.ytrain = torch.tensor(ytrain, dtype=torch.long)
        self.yvalidation = torch.tensor(yvalidation, dtype=torch.long)
        self.ytest = torch.tensor(ytest, dtype=torch.long)
        self.scaler = scaler
        self.features = features

        return self.xtrain, self.xvalidation, self.xtest, self.ytrain, self.yvalidation, self.ytest, self.scaler, self.features

    def train(self):
        # https://scikit-learn.org/stable/modules/naive_bayes.html
        self.load_data()
        self.bayes.fit(self.xtrain,self.ytrain)

    def predict(self,test):
        pred = self.bayes.predict(test)
        #print(pred)
        return pred

    def evaluate_model(self):
        pred = self.predict(self.xtest)
        output= classification_report(self.ytest.numpy(),pred)
        # variables for confusion matrix
        tp =0
        fp= 0
        fn=0
        tn=0
        for i in range(len(pred)):
            if pred[i] == 1:
                if self.ytest[i] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if self.ytest[i] == 1:
                    fn+=1
                else:
                    tn+=1
        print("fn", fn ," fp ", fp , " tn " , tn , " tp ", tp)
        print(output)

bayes = BayesClassifier()
bayes.train()
bayes.predict(b.xtest)
bayes.evaluate_model()
