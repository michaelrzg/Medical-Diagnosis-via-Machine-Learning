import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the data the same way as the supervised data but without labels
def load_and_prepare_data():
    # Load the dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target #useful for comparing the clusters to the actual labels

    # note that this alg only loads the X values, not the Y values (no labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# train a k-means model (creates clusters and matches each datapoint to the cluster it is closest to)
def train_kmeans_model(X_train, n_clusters=2):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    # Train the model
    kmeans.fit(X_train)
    return kmeans

def evaluate_kmeans_model(kmeans, X_test, y_test):
    # Predict the clusters for the test data
    cluster_labels = kmeans.predict(X_test)
    assigned_clusers = assign_clusters_to_labels(kmeans, X_test, y_test)
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
    #print(cluster_labels, y_test, assigned_clusers)

# goes through each cluster and finds the majority value (0 or 1) then assigns that cluster that valeu
def assign_clusters_to_labels(kmeans, X_train, y_train):

    clusters = kmeans.predict(X_train)
    cluster_labels = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_true_labels = y_train[cluster_indices]
        # Find the most frequent label in this cluster
        if cluster_true_labels.size > 0:
            most_common_label = np.bincount(cluster_true_labels).argmax()
        else:
            most_common_label = 0
        cluster_labels[cluster_id] = most_common_label
    return cluster_labels

def predict_with_kmeans(kmeans, cluster_labels_map,  data):

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
    # Load and prepare the data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()

    # Train the K-Means model
    n_clusters = 2  # benign or malig
    kmeans_model = train_kmeans_model(X_train, n_clusters=n_clusters)

    # Evaluate the model
    evaluate_kmeans_model(kmeans_model, X_test, y_test)

    # Assign labels to clusters
    cluster_labels_map = assign_clusters_to_labels(kmeans_model, X_train, y_train)
    print(f"Cluster to label mapping: {cluster_labels_map}")
