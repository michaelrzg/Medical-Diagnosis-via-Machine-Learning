# Michael Rizig
# Unsupervised Learning alg (kmeans) to predict breast cancer dataset
# 4/10/25
# Professor Alexiou

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class Unsupervised:
    def __init__(self):
        self.load_data()
        self.train_kmeans_model()
        self.assigned_clusters =  self.assign_clusters_to_labels()
    # load the data the same way as the supervised data but without labels
    def load_data(self):
        # Load the dataset
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target #useful for comparing the clusters to the actual labels

        # note that this alg only loads the X values, not the Y values (no labels)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        self.xtrain = scaler.fit_transform(self.xtrain)
        self.xtest = scaler.transform(self.xtest)

        return self.xtrain, self.xtest, self.ytrain, self.ytest, scaler

    # train a k-means model (creates clusters and matches each datapoint to the cluster it is closest to)
    def train_kmeans_model(self, n_clusters=2):

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        # Train the model
        self.kmeans.fit(self.xtrain)
        return self.kmeans

    def evaluate(self):
        # Predict the clusters for the test data
        cluster_labels =self.kmeans.predict(self.xtest)
        if self.assigned_clusters[0] == 1: # if the labels are flipped, flip the labels of each
            for i in range(len(cluster_labels)):
                if cluster_labels[i] == 1:
                    cluster_labels[i] = 0
                else :
                    cluster_labels[i] = 1
        # generate variables for confusion matrix (same as before)
        tp =0
        fp= 0
        fn=0
        tn=0
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == 1:
                if self.ytest[i] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if self.ytest[i] == 1:
                    fn+=1
                else:
                    tn+=1
        #print("fn", fn ," fp ", fp , " tn " , tn , " tp ", tp)
        #print(cluster_labels, self.ytest, assigned_clusers)
        return classification_report(self.ytest,cluster_labels)

    # goes through each cluster and finds the majority value (0 or 1) then assigns that cluster that valeu
    def assign_clusters_to_labels(self):

        clusters = self.kmeans.predict(self.xtrain)
        cluster_labels = {}
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_true_labels = self.ytrain[cluster_indices]
            # Find the most frequent label in this cluster
            if cluster_true_labels.size > 0:
                most_common_label = np.bincount(cluster_true_labels).argmax()
            else:
                most_common_label = 0
            cluster_labels[cluster_id] = most_common_label
        return cluster_labels

    def predict(self,  data):
        pred = self.kmeans.predict(data)
        if self.assigned_clusters[0] == 1: # if the labels are flipped, flip the labels of each
            for i in range(len(pred)):
                if pred[i] == 1:
                    pred[i] = 0
                else :
                    pred[i] = 1
        return pred

if __name__ == "__main__":
    unsupervised = Unsupervised()
    # Evaluate the model
    print(unsupervised.evaluate())
    # Assign labels to clusters

    #print(f"Cluster to label mapping: {unsupervised.assign_clusters_to_labels()}")
