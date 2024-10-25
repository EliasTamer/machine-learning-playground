import numpy as np

def find_closest_centroids(X, centroids):
    """
    This function takes the data matrix X and the locations of all centroids inside centroids.
    
    It should output a one-dimensional array idx (which has the same number of elements as X)
    that holds the index of the closest centroid.
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """
    K = centroids.shape[0]
    
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(len(idx)):
        
        distance = []
        
        for j in range(K):
            distance_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(distance_ij)
            
        idx[i] = np.argmin(distance)
        
    return idx



def compute_centroids(X, idx, K):
    """
    Given assignments of every point to a centroid, the second phase of the algorithm recomputes,
    for each centroid, the mean of the points that were assigned to it.
    
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    m, n = X.shape
    
    centroids = np.zeros((K, n))
    
    for k in range(K):        
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)    
    
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)

    for i in range(max_iters):
        
        print("K-Means iteration %d/%d" % (i, max_iters-1))

        idx = find_closest_centroids(X, centroids)
        
        centroids = compute_centroids(X, idx, K)
    return centroids, idx