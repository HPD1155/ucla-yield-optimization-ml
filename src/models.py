import numpy as np

# Stand
class StandardScaler:
    def __init__(self):
        self.mean = 0 # Set mean to 0
        self.std = 0 # Set standard deviation to 0
    
    def fit(self, X):
        # Get the mean of X Series
        self.mean = np.mean(X, axis=0)
        # Get standard deviation of X series
        self.std = np.sqrt(np.sum((X - self.mean)**2, axis=0) / len(X)) 
        return self.mean, self.std
    
    def transform(self, X):
        # Scale X Series based on given std and mean
        return np.divide((X - self.mean), self.std, out=np.zeros_like(X - self.mean), where=self.std!=0)
    

    def fit_transform(self, X):
        # Combines two functions into one
        self.mean, self.std = self.fit(X)
        return self.transform(X)

# KMeans Algorithm
class KMeans:
    def __init__(self, n_clusters=4, max_iter=100):
        self.k = n_clusters
        self.max_iter = max_iter
        self.centroids = None # will be numpy array storing positions of centroids (k, n_features)
        self.clusters = None # will be numpy array storing instances of elements assigned to each cluster (n_samples,)
    
    def _euclid_distance(self, X1, X2):
        return np.sqrt(np.sum(np.square(X1 - X2), axis=1))
    
    def initalize_centroids(self, X):
        self.centroids = X[np.random.choice(X.shape[0], size=self.k)] # Initialize random positions for each centroid based on dimensions of X
        return self.centroids

    def assign_clusters(self, X):
        distance = np.zeros((X.shape[0], self.k))
        for k in range(self.k): # iterate k times to check distances
            # Vector operation on each row where for each x point, it calculates its distance from the centroid
            distance[:, k] = self._euclid_distance(X, self.centroids[k])
        self.clusters = np.argmin(distance, axis=1) # Finds the index of the closest centroid from x
        return self.clusters        

    def update_centroids(self, X):
        # Find the mean of all points in centroid_n's cluster, position centroid to be in the middle of that
        for k in range(self.k):
            if np.any(self.clusters == k):
                self.centroids[k] = np.mean(X[self.clusters == k], axis=0)
            else:
                self.centroids[k] = X[np.random.choice(X.shape[0])]
        
        return self.centroids
    
    def update(self, X):
        self.assign_clusters(X)
        self.update_centroids(X)

    def fit(self, X):
        self.initalize_centroids(X)
        for i in range(self.max_iter):
            self.update(X)
        return self.clusters
    
    def WCSS(self, X):
        
        return np.sum(np.square(X - self.centroids[self.clusters]))

    def predict(self, X):
        distance = np.zeros((X.shape[0], self.k))
        for k in range(self.k): # iterate k times to check distances
            # Vector operation on each row where for each x point, it calculates its distance from the centroid
            distance[:, k] = self._euclid_distance(X, self.centroids[k])
        pred = np.argmin(distance, axis=1) # Finds the index of the closest centroid from x
        return pred

if __name__ == "__main__":
    X = np.random.rand(10, 2)
    kmeans = KMeans(3, 100)
    print(f"Initialized Centroids: {kmeans.initalize_centroids(X)}")
    print("Clusters: \n")
    print(kmeans.fit(X))
    print(kmeans.WCSS(X))