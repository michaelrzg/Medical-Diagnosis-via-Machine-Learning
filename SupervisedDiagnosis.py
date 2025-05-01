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


def load_and_prepare_data():
    # Load the dataset
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    # grab features

    features = list(X.columns)
    # Split the data into train, test, adn validation sets
    # lines 19 and 20 borrowed from sklearn website
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 of 0.8 is 0.2

    # normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, features


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

# divde and conquer reduce:

def divide_and_conquer_feature_selection(X_train, y_train, X_test, y_test, features, target_num_features):
    if len(features) <= target_num_features:
        return features

    # Divide the features into two
    mid = len(features) // 2
    left = features[:mid]
    right = features[mid:]

    # Evaluate
    aleft = helper(X_train, y_train, X_test, y_test, left)
    aright = helper(X_train, y_train, X_test, y_test, right)

    # Keep the better half
    selected = left if aleft > aright else right

    # Recurse
    return divide_and_conquer_feature_selection(X_train, y_train, X_test, y_test, selected, target_num_features)

def helper(X_train, y_train, X_test, y_test, feature_subset):
    model = LogisticRegression(max_iter=10000, solver='liblinear')
    model.fit(X_train[feature_subset], y_train)
    y_pred = model.predict(X_test[feature_subset])
    return accuracy_score(y_test, y_pred)



def train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    val_accuracies = []
    for epoch in range(epochs):
        # Forward propagation
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train.float().view(-1, 1))
        # Backward propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        with torch.no_grad():
            y_pred_val = model(X_val)
            y_pred_val_binary = (y_pred_val > 0.5).long()  # Threshold at 0.5
            val_accuracy = accuracy_score(y_val.numpy(), y_pred_val_binary.numpy())
            val_accuracies.append(val_accuracy)


    return val_accuracies

def predict(model, scaler, data):
    # Convert the input data to a numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data).reshape(1, -1)  # Reshape to (1, num_features)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
          data = data.reshape(1, -1)
    else:
        raise TypeError("Input data must be a list or a numpy array.")

    # Scale the data using the scaler fitted on the training data
    data_scaled = scaler.transform(data)
    # Convert the scaled data to a tensor
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    # Make the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        y_pred_prob = model(data_tensor)
        y_pred_binary = (y_pred_prob > 0.5).long()  # Threshold at 0.5

    return y_pred_binary.item()  # Return the prediction as a Python integer

def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_test_binary = (y_pred_test > 0.5).long()  # Threshold at 0.5
        test_accuracy = accuracy_score(y_test.numpy(), y_pred_test_binary.numpy())
    # variables for confusion matrix
    tp =0
    fp= 0
    fn=0
    tn=0
    for i in range(len(y_pred_test_binary)):
        if y_pred_test_binary[i] == 1:
            if y_test[i] == 1:
                tp+=1
            else:
                fp+=1
        else:
            if y_test[i] == 1:
                fn+=1
            else:
                tn+=1
    print("fn", fn ," fp ", fp , " tn " , tn , " tp ", tp)
    print(classification_report(y_test.numpy(), y_pred_test_binary.numpy()))

if __name__ == "__main__":
    # Load and prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, features = load_and_prepare_data()

    # Define the model
    input_size = X_train.shape[1]  # Number of features
    model = BinaryClassifier(input_size)

    # Train the model
    val_accuracies = train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
