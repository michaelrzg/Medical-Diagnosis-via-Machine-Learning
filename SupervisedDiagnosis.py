# Michael Rizig
# Supervised Learning alg to predict breast cancer dataset
# 4/10/25
# Professor Alexiou

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
# 2. Define the Model
class BinaryClassifier(nn.Module):

    def __init__(self, input_size):

        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # layer 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)       # layer 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)       #  layer 3 (output layer)
        self.sigmoid = nn.Sigmoid()       # map to value between 0 and 1

    def forward(self, x):
        # forward propigate through network
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class Supervised:
    def __init__(self):
        self.classifier = BinaryClassifier(30)
        self.load_data()
        self.train_model()
    def load_data(self):
        # Load the dataset
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = cancer.target
        # grab features

        self.features = list(X.columns)
        # Split the data into train, test, adn validation sets
        # lines 19 and 20 borrowed from sklearn website
        xtrain, xtest, ytrain, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xtrain, xvalidation, ytrain, yvalidation = train_test_split(xtrain, ytrain, test_size=0.25, random_state=42) # 0.25 of 0.8 is 0.2

        # normalize the data
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler
        self.scaler = StandardScaler()
        xtrain = self.scaler.fit_transform(xtrain)
        xvalidation = self.scaler.transform(xvalidation)
        xtest = self.scaler.transform(xtest)

        # Convert data to tensors
        self.xtrain = torch.tensor(xtrain, dtype=torch.float32)
        self.xvalidation = torch.tensor(xvalidation, dtype=torch.float32)
        self.xtest = torch.tensor(xtest, dtype=torch.float32)
        self.ytrain = torch.tensor(ytrain, dtype=torch.long)
        self.yvalidation = torch.tensor(yvalidation, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        return self.xtrain, self.xvalidation, self.xtest, self.ytrain, self.yvalidation, y_test, self.scaler, self.features

    # divde and conquer reduce:

    def divide_and_conquer_feature_selection(self,xtrain, ytrain, xtest, ytest, features, num_features):
        if len(features) <= num_features:
            return features

        # Divide the features into two
        mid = len(features) // 2
        left = features[:mid]
        right = features[mid:]

        # Evaluate
        aleft = self.helper(xtrain, ytrain, xtest, ytest, left)
        aright = self.helper(xtrain, ytrain, xtest, ytest, right)

        # Keep the better half
        selected = left if aleft > aright else right

        # Recurse
        return self.divide_and_conquer_feature_selection(xtrain, ytrain, xtest, ytest, selected, num_features)

    def helper(self,xtrain, ytrain, xtest, ytest, feature_subset):
        model = LogisticRegression(max_iter=10000, solver='liblinear')
        model.fit(xtrain[feature_subset], ytrain)
        y_pred = model.predict(xtest[feature_subset])
        return accuracy_score(ytest, y_pred)



    def train_model(self, epochs=100, learning_rate=0.001):

        lossc = nn.BCELoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)

        # Train the model
        val_accuracies = []
        for epoch in range(epochs):
            # Forward propagation
            pred = self.classifier(self.xtrain)
            loss = lossc(pred, self.ytrain.float().view(-1, 1))
            # Backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate on validation set
            with torch.no_grad():
                y_pred_val = self.classifier(self.xvalidation)
                y_pred_val_binary = (y_pred_val > 0.5).long()  # Threshold at 0.5
                val_accuracy = accuracy_score(self.yvalidation.numpy(), y_pred_val_binary.numpy())
                val_accuracies.append(val_accuracy)


        return val_accuracies

    def predict(self, data):
        pred = self.classifier(data)
        pred = (pred>0.5).int().tolist()
        return sum(pred,[])

    def evaluate_model(self):
        # Evaluate the model on the test set
        with torch.no_grad():
            y_pred_test = self.classifier(self.xtest)
            y_pred_test_binary = (y_pred_test > 0.5).long()  # Threshold at 0.5
            #test_accuracy = accuracy_score(self.y_test, y_pred_test_binary.numpy())
        # variables for confusion matrix
        tp =0
        fp= 0
        fn=0
        tn=0
        for i in range(len(y_pred_test_binary)):
            if y_pred_test_binary[i] == 1:
                if self.y_test[i] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if self.y_test[i] == 1:
                    fn+=1
                else:
                    tn+=1
        #print("fn", fn ," fp ", fp , " tn " , tn , " tp ", tp)
        return classification_report(self.y_test, y_pred_test_binary.numpy())

if __name__ == "__main__":

    supervised = Supervised()
    supervised.predict(supervised.xtest)
    # Evaluate the model
    print(supervised.evaluate_model())
