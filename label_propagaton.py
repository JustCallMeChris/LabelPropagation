import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors.unsupervised import NearestNeighbors
from time import time as time

#     Parameters
#     ----------
#     kernel : {'knn', 'rbf', 'dbscan'}
#         String identifier for kernel function to use.
#         dbscan: use the result as a affinity matrix for the propagation. 
#     gamma : float
#         Parameter for rbf kernel
#     n_neighbors : integer > 1
#         Parameter for knn kernel
#         first neighbors is self -> must be over 1
#     unlabeledValue : object
#         the unknown label -> this will be override
#     eps: float
#         Value to define distance of epsilon neighborhood epsilon.
#     minPtsMin: integer
#         minimum amount of points to be in the neighborhood of a
#         data point p for p to be recognized as a core point.
#
class LabelPropagation:
    def __init__ (self, kernel=None, gamma=10, unlabeledValue=0 , eps=1, minPts=1, neighbors=2):
        if neighbors < 2:
            raise
        self.kernel         = kernel
        self.gamma          = gamma
        self.unlabeledValue = unlabeledValue 
        self.eps            = eps
        self.minPts         = minPts
        self.naighbors      = neighbors
        
#     Fit a semi-supervised label propagation model based
#     All the input data is provided matrix X (labeled and unlabeled)
#     and corresponding label matrix y with a dedicated marker value for
#     unlabeled samples.
#     Parameters
#     ----------
#     X : array-like, shape = [n_samples, n_features]
#         A {n_samples by n_samples} size matrix will be created from this
#     y : array_like, shape = [n_samples]
#         n_labeled_samples
#         All unlabeled samples will be transductively assigned labels
#     Returns
#     -------
#     self : returns an instance of self.
    def fit(self, X, y):
        t = time()  # get labels for test data
        # build the graph result is the affinity matrix
        if self.kernel is 'dbscan' or self.kernel is None:
            affinity_matrix = self.dbscan(X, self.eps, self.minPts)
        # it is possible to use other kernels -> as parameter
        elif self.kernel is 'rbf':
            affinity_matrix = rbf_kernel(X, X, gamma=self.gamma)
        elif self.kernel is 'knn':
            affinity_matrix = NearestNeighbors(self.naighbors).fit(X).kneighbors_graph(X, self.naighbors).toarray()
        else:
            raise
        print( "praph(%s) time %2.3fms"%(self.kernel, (time() - t) *1000))
        if affinity_matrix.max() == 0 :
            print("no affinity matrix found")
            return y
        
        degree_martix   = np.diag(affinity_matrix.sum(axis=0))
        affinity_matrix = np.matrix(affinity_matrix)
        
        try:
            inserve_degree_matrix = np.linalg.inv(degree_martix)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.args:
                # use a pseudo inverse if it's not possible to make a normal of the degree matrix
                inserve_degree_matrix =  np.linalg.pinv(degree_martix)
            else:
                raise
            
        matrix = inserve_degree_matrix * affinity_matrix
        # split labels in different vectors to calculate the propagation for the separate label
        labels = np.unique(y)
        labels = [x for x in labels if x != self.unlabeledValue]
        # init the yn1 and y0
        y0  = [[1 if (x == l) else 0 for x in y] for l in labels]
        yn1 = y0
        # function to set the probability to 1 if it was labeled in the source
        toOrgLabels      = np.vectorize(lambda x, y : 1 if y == 1 else x , otypes=[np.int0])
        # function to set the index's of the source labeled
        toOrgLabelsIndex = np.vectorize(lambda x, y, z : z if y == 1 else x , otypes=[np.int0])
        lastLabels       = np.argmax(y0, axis=0)
        while True:
            yn1 = yn1 * matrix
            #first matrix to labels
            ynLablesIndex = np.argmax(yn1, axis=0)
            # row-normalize
            yn1 /= yn1.max()
            yn1 = toOrgLabels(yn1, y0)
            for x in y0:
                ynLablesIndex = toOrgLabelsIndex(ynLablesIndex, x, y0.index(x))
            #second original labels to result
            if np.array_equiv(ynLablesIndex, lastLabels):
                break
            lastLabels = ynLablesIndex
        # result is the index of the labels -> cast index to the given labels
        toLabeles = np.vectorize(lambda x : labels[x])
        return np.array(toLabeles(lastLabels))[0]

    def dbscan(self, data, eps, minPts):
        # Compute distances of data points
        distances = self.computeDistances(data)
        
        # Dimensions of data matrix
        numPoints = data.shape[0]
        
        # Current cluster
        # This is used to add columns to memberships matrix.
        currentCluster = -1
        
        # Matrix to store membership degrees of points.
        # Initiated to -1 just to reserve an index in the matrix.
        memberships = [[-1] for i in range(numPoints)]
        
        # Array to store if a point is already visited.
        # Visited indicates we already computed the
        # eps-neighborhood once for core points.
        visited = [False] * numPoints
        
        for i in range(numPoints):
            # If the current data point was already visited before,
            # stop here.
            if visited[i]:
                continue
            
            # Compute eps-neighborhood of current data point
            neighbors = self.computeNeighbors(distances, i, eps)
            
            # If this data point is a core point, treat it appropriately.
            if len(neighbors) >= minPts:
                # Mark current data point as visited
                visited[i] = True
                
                # Increment cluster id
                currentCluster += 1
                
                # Add a column to memberships if necessary
                # Might be done more efficiently.
                if currentCluster > 0:
                    for row in memberships:
                        row.append(-1)
                
                # Grow this cluster
                self.expandCluster(i, neighbors, eps, minPts, visited, memberships, distances, currentCluster)
            
        # Compute crisp clustering out of membership matrix
        # -1 is noise, everything else is a cluster index
        clustering = []
        for i in range(numPoints):
            cluster = -1
            maxMembership = -1
            for j in range(currentCluster+1):
                currentMembership = memberships[i][j]
                if currentMembership > maxMembership:
                    cluster = j
                    maxMembership = currentMembership
            clustering.append(cluster)
                
        # Ininit similarity matrix with zero distance
        similarity = np.zeros((numPoints,numPoints))
        for i in range(numPoints):
            neighbors = self.computeNeighbors(distances, i, eps)
            for k in neighbors:
                if k > i and clustering[i] == clustering[k]:
                    similarity[i][k] = distances[i][k]
                    similarity[k][i] = distances[i][k]
    #                 print(distances[i][k])
      
        #Plotter.visualizeClustering(data, clustering, eps, minPts) 
        return similarity
    
    #
    # This function grows a cluster such that every data point of the cluster currentCluster will be found.
    #
    # Parameters are:
    # point:            First processed core point of this cluster
    # neighbors:        Epsilon neighborhood of point (this is a set)
    # eps:              Value to define distance of epsilon neighborhood epsilon.
    # minPtsMin:        minimum amount of points to be in the neighborhood of a
    #                   data point p for p to be recognized as a core point.
    # visited:          Array of flags to show if the the epsilon neighborhood has already
    #                   been computed for each of the data points.
    # memberships:      Matrix to store membership degrees of points.
    # distances:        numpy.ndarray that is an upper triangular matrix with diagonal 0-entries.
    # currentCluster:   Index of the currently processed cluster
    def expandCluster(self, point, neighbors, eps, minPts, visited, memberships, distances, currentCluster):
        # set of border points of this cluster
        borderPoints = set()
        # Set of core points of this cluster
        corePoints = set()
        # Add data point to the current cluster with fuzzy membership degree
        memberships[point][currentCluster] = self.computeMembershipDegree(len(neighbors), minPts)
        # Add core point to set of core points
        corePoints.add(point)
        
        # As long as neighbors is not empty
        while neighbors:
            i = neighbors.pop()
            # If this neighbor is not already visited
            # and not a border point.
            if not visited[i] and not (i in borderPoints):
                # Compute neighbors of current neighbor
                neighbors2 = self.computeNeighbors(distances, i, eps)
                
                # Core point
                if len(neighbors2) >= minPts:
                    # Mark current neighbor as visited
                    visited[i] = True
                    # Add core point to set of core points
                    corePoints.add(i)
                    # Take neighbors into consideration
                    neighbors = neighbors.union(neighbors2)
                    # Assign membership degree to this core point
                    memberships[i][currentCluster] = self.computeMembershipDegree(len(neighbors2), minPts)
                # Border point
                else:
                    # Take care: Don't set this point to be visited!
                    borderPoints.add(i)
        
        # Compute membership degrees of this cluster's border points
        # to introduce the desired fuzzy aspect.
        while borderPoints:
            i = borderPoints.pop()
            # Compute neighbors of this data point.
            # This might happen more than once for border points.
            neighbors2 = self.computeNeighbors(distances, i, eps)
            # Which neighbors are core points of this cluster?
            coreNeighbors = neighbors2.intersection(corePoints)
            # Which core point has the biggest membership degree?
            biggestMembership = -1
            while coreNeighbors:
                j = coreNeighbors.pop()
                currentMembership = memberships[j][currentCluster]
                if biggestMembership < currentMembership:
                    biggestMembership = currentMembership
            
            # Set membership degree of current border point to
            # the maximum membership degree of its core neighbors
            # of this cluster.
            memberships[i][currentCluster] = biggestMembership
        
        return
    
    # Function to compute the eps-neighborhood of a data point as a set of indizes.
    #
    # Parameters are:
    # distances:    numpy.ndarray that is an upper triangular matrix with diagonal 0-entries.
    # point:        Index in distance matrix of data point to compute the neighborhood for.
    # eps:          Value to define distance of epsilon neighborhood epsilon.
    #
    # Returns set of neighbor points as indizes into distance matrix.
    def computeNeighbors(self, distances, point, eps):
        neighbors = set()
        
        # Look at first part of distances
        for i in range(point):
            if distances[i][point] <= eps:
                neighbors.add(i)
        
        # Look at second part of distances
        numPoints = distances.shape[1]
        # We insert the point itself as it's own neighbor.
        for i in range(point,numPoints):
            if distances[point][i] <= eps:
                neighbors.add(i)
        
        return neighbors
    
        
    # Function to calculate fuzzy membership degrees.
    #
    # Parameters are:
    # numNeighbors:    Number of neighbors of a data point.
    # minPtsMin:       minimum amount of points to be in the neighborhood of a
    #                  data point p for p to be recognized as a core point.
    # mintPtsMax:      maximum amount a points in the neighborhood of a data point
    #                  which leads to maximum membership degree of 1 for points with
    #                  at least minPtsMax neighbors. This parameter helps to recognize
    #                  more degrees of density. Thats's why it is recommended to use
    #                  big values.
    def computeMembershipDegree(self, numNeighbors, minPts):
        if numNeighbors >= minPts:
            return 1
        if numNeighbors < minPts:
            return 0;
    
    # This function computes the Euclidean distance of a matrix of data points.
    # Parameters are:
    # data:        numpy.ndarray of data points.
    #
    # Returns an upper triangular matrix (with diagonal 0-values),
    # that is filled with numbers/distances.
    def computeDistances(self, data):
    
        lenArrayOfPoints = len(data)
        dimension = len(data[0])
        
        distanceMatrix = []
        
        # Rows
        for i in range(lenArrayOfPoints): 
            
            distanceCollector = [[]]
            # Columns
            for j in range(lenArrayOfPoints): 
                if i >= j: 
                    # Fills lower triangular matrix with 0
                    distanceCollector[0].extend([0])
                else:
                    euclideanDistanceAddition = 0
                    
                    # Computes euclidean distance
                    for k in range(dimension):
                        euclideanDistanceAddition = euclideanDistanceAddition + (data[i][k]-data[j][k])**2
                        
                    euclideanDistance = euclideanDistanceAddition**(1/2.0)   
                    distanceCollector[0].extend([euclideanDistance])
            
            # Adds row to array of distance matrix        
            distanceMatrix.extend(distanceCollector)
        # Distance Matrix as NumpyArray  
        distanceMatrix = np.array(distanceMatrix)  
    
        return distanceMatrix
