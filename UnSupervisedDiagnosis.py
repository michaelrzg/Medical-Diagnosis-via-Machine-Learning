import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class Unsupervised:
    # load the data the same way as the supervised data but without labels
    def load_data(self):
        # Load the dataset
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target #useful for comparing the clusters to the actual labels

        # note that this alg only loads the X values, not the Y values (no labels)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

        return xtrain, xtest, ytrain, ytest, scaler

    # train a k-means model (creates clusters and matches each datapoint to the cluster it is closest to)
    def train_kmeans_model(self,xtrain, n_clusters=2):

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        # Train the model
        kmeans.fit(xtrain)
        return kmeans

    def evaluate_kmeans_model(self,kmeans, xtest, ytest):
        # Predict the clusters for the test data
        cluster_labels = kmeans.predict(xtest)
        assigned_clusers = self.assign_clusters_to_labels(kmeans, xtest, ytest)
        if assigned_clusers[0] == 1: # if the labels are flipped, flip the labels of each
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
                if ytest[i] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if ytest[i] == 1:
                    fn+=1
                else:
                    tn+=1
        print("fn", fn ," fp ", fp , " tn " , tn , " tp ", tp)
        print(cluster_labels, ytest, assigned_clusers)

    # goes through each cluster and finds the majority value (0 or 1) then assigns that cluster that valeu
    def assign_clusters_to_labels(self,kmeans, xtrain, ytrain):

        clusters = kmeans.predict(xtrain)
        cluster_labels = {}
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_true_labels = ytrain[cluster_indices]
            # Find the most frequent label in this cluster
            if cluster_true_labels.size > 0:
                most_common_label = np.bincount(cluster_true_labels).argmax()
            else:
                most_common_label = 0
            cluster_labels[cluster_id] = most_common_label
        return cluster_labels

    def predict_with_kmeans(self,kmeans, cluster_labels_map,  data):

        # Convert the input data to a numpy array if it's a list (same as before)
        if isinstance(data, list):
            data = np.array(data).reshape(1, -1)  # Reshape to (1, num_features)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(1, -1)
        else:
            raise TypeError("Input data must be a list or a numpy array.")

        # Predict the cluster for the new data point
        cluster = kmeans.predict(data)[0]
        # Get the label for that cluster
        predicted_label = cluster_labels_map[cluster]
        return predicted_label

if __name__ == "__main__":
    unsupervised = Unsupervised()
    # Load and prepare the data
    xtrain, xtest, ytrain, ytest, scaler = unsupervised.load_data()

    # Train the K-Means model
    n_clusters = 2  # benign or malig
    kmeans_model = unsupervised.train_kmeans_model(xtrain, n_clusters=n_clusters)

    # Evaluate the model
    unsupervised.evaluate_kmeans_model(kmeans_model, xtest, ytest)

    # Assign labels to clusters
    cluster_labels_map = unsupervised.assign_clusters_to_labels(kmeans_model, xtrain, ytrain)
    print(f"Cluster to label mapping: {cluster_labels_map}")
