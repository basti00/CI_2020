#Filename: HW5_skeleton.py
#Author: Christian Knoll
#Edited: May 2020

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import datasets

#--------------------------------------------------------------------------------
# Assignment 5
def main():
    #------------------------
    # 0) Get the input
    ## (a) load the modified iris data
    data, labels, feature_names = load_iris_data()

    ## (b) construct the datasets
    x_2dim = data[:, [0,2]]
    x_4dim = data

    #TODO: implement PCA
    #x_2dim_pca = PCA(data,nr_dimensions=2,whitening=False)

    ## (c) visually inspect the data with the provided function (see example below)
    plot_iris_data(x_2dim,labels, feature_names[0], feature_names[2], "Iris Dataset")

    #------------------------
    # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm
    scenario = 1
    dim = 2
    nr_components = 3

    #TODO set parameters
    tol = 0.01  # tolerance
    max_iter = 100  # maximum iterations for GN
    #nr_components = ... #n number of components

    #TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    EM(x_2dim,nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    #initial_centers = init_k_means(dimension = dim, nr_clusters=nr_components, scenario=scenario, X=x_2dim)
    #k_means(x_2dim, nr_components, initial_centers, max_iter, tol, labels, feature_names)

    #TODO visualize your results

    #------------------------
    # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm
    scenario = 2
    dim = 4
    nr_components = 3

    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    #nr_components = ... #n number of components

    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    #initial_centers = init_k_means(dimension = dim, nr_cluster=nr_components, scenario=scenario)
    #... = k_means(x_2dim,nr_components, initial_centers, max_iter, tol)

    #TODO: visualize your results by looking at the same slice as in 1)


    #------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data
    scenario = 3
    dim = 2
    nr_components = 3

    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    #nr_components = ... #n number of components

    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_2dim_pca, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    #initial_centers = init_k_means(dimension = dim, nr_cluster=nr_components, scenario=scenario)
    #... = k_means(x_2dim_pca, nr_components, initial_centers, max_iter, tol)

    #TODO: visualize your results
    #TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)

    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def init_EM(dimension=2,nr_components=3, scenario=None, X=None):
    """ initializes the EM algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... (optional) parameter that allows to further specify the settings, scalar
        X... (optional) samples that may be used for proper inititalization, nr_samples x dimension(D)
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""
    
    # TODO choose suitable initial values for each scenario
    
    alpha_0 = np.ones((1, nr_components))
    mean_0 = np.ones((dimension, nr_components))
    cov_0 = np.ones((dimension, dimension, nr_components))
    print(alpha_0.shape)
    print(mean_0.shape)
    print(cov_0.shape)

    return (alpha_0, mean_0, cov_0)
#--------------------------------------------------------------------------------
def EM(X,K,alpha_0,mean_0,cov_0, max_iter, tol):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension
    D = X.shape[1]
    assert D == mean_0.shape[0]
    #TODO: iteratively compute the posterior and update the parameters
    
    r = np.zeros((len(X), K))

    log_likelihood = []

    for i in range(max_iter):
        for n in range(len(X)):
            for k in range(K):
                r_nenner = 0
                for k_ in range(K):
                    r_nenner += alpha_0[k_] * likelihood_multivariate_normal(X[n], mean_0.T[k_], cov_0.T[k_])

                r[k][n] = alpha_0[k] * likelihood_multivariate_normal(X[n], mean_0[k], cov_0[k]) / r_nenner

        N = 0
        N_k = 0
        for k in range(K):
            #calc alpha_0 new
            mean_temp = 0
            
            for n in range(len(X)):
                mean_temp += r[k][n]*X[n]
                N_k += r[k][n]
                N = += r[k][n]


            mean_0[k] = mean_temp / N_k

            cov_temp = 0
            for n in range(len(X)):
                trans = X[n] * mean_0[k]
                cov_temp += r[k][n] * (X[n] - mean_0[k]) * trans.T

            cov_0[k] = cov_temp / N_k

        alpha_0 = N_k / N

        log_likelihood_it = 0

        for n in range(len(X)):
            for k in range(K):
                log_likelihood_it += likelihood_multivariate_normal(X[n], mean_0[k], cov_0[k])
        
        log_likelihood.append(log_likelihood_it)


            




    #TODO: classify all samples after convergence
    pass
#--------------------------------------------------------------------------------
def init_k_means(dimension=None, nr_clusters=None, scenario=None, X=None):
    """ initializes the k_means algorithm
    Input:
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... (optional) parameter that allows to further specify the settings, scalar
        X... (optional) samples that may be used for proper inititalization, nr_samples x dimension(D)
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    #TODO: choose suitable inital values for each scenario

    if (scenario == 1):
        return X[np.random.choice(X.shape[0], nr_clusters, replace=False)].T

#--------------------------------------------------------------------------------
def k_means(X,K, centers_0, max_iter, tol, labels, feature_names):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    assert D == centers_0.shape[0]
    #TODO: iteratively update the cluster centers

    #indices of closest centers for each point
    nearest_centers = np.zeros(X.shape[0], dtype=np.int)
    centers = centers_0.T
    cumulative_distance = np.ndarray((0,1))

    for i in range(max_iter):
        distance_sum = 0

        for x_index, x in enumerate(X):
            # find closest center
            min_dist = np.inf
            for center_index, center in enumerate(centers):
                dist = np.linalg.norm(x-center)
                if dist < min_dist:
                    min_dist = dist
                    distance_sum+=dist
                    nearest_centers[x_index] = center_index

        if i == 0 or np.abs(distance_sum - cumulative_distance[i-1]) > tol:
            cumulative_distance = np.append(cumulative_distance, distance_sum)
            # set new centers
            centers = np.zeros((K, X.shape[1]))
            for x_index, center_index in enumerate(nearest_centers):
                centers[center_index] += 1/len([x for x in nearest_centers if x == center_index]) * X[x_index]
        else:
            print("Done")
            break
        
        plt.scatter(X[nearest_centers==0,0], X[nearest_centers==0,1], label='Iris-Setosa')
        plt.scatter(X[nearest_centers==1,0], X[nearest_centers==1,1], label='Iris-Versicolor')
        plt.scatter(X[nearest_centers==2,0], X[nearest_centers==2,1], label='Iris-Virgnica')
        plt.scatter(centers.T[0], centers.T[1], label='Centers')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[2])
        plt.title("test")
        plt.legend()
        plt.show()

        print(i, nearest_centers)

    return centers.T, cumulative_distance, nearest_centers


    #TODO: classify all samples after convergence
    return
#--------------------------------------------------------------------------------
def PCA(data,nr_dimensions=None, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening

    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    #TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components


    #TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input:
    Returns:
        X... samples, 150x4
        Y... labels, 150x1
        feature_names... name of the data columns"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100,2] =  iris.data[50:100,2]-0.25
    Y = iris.target
    return X,Y, iris.feature_names
#--------------------------------------------------------------------------------
def plot_iris_data(data, labels, x_axis, y_axis, title):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1
        x_axis... label for the x_axis
        y_axis... label for the y_axis
        title...  title of the plot"""

    plt.scatter(data[labels==0,0], data[labels==0,1], label='Iris-Setosa')
    plt.scatter(data[labels==1,0], data[labels==1,1], label='Iris-Versicolor')
    plt.scatter(data[labels==2,0], data[labels==2,1], label='Iris-Virgnica')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    plt.show()
#--------------------------------------------------------------------------------
def likelihood_multivariate_normal(X, mean, cov, log=False):
   """Returns the likelihood of X for multivariate (d-dimensional) Gaussian
   specified with mu and cov.

   X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
   mean ... mean -- [mu_1, mu_2,...,mu_d]
   cov ... covariance matrix -- np.array with (d x d)
   log ... False for likelihood, true for log-likelihood
   """

   print(mean)
   print(cov)
   dist = multivariate_normal(mean, cov)
   if log is False:
       P = dist.pdf(X)
   elif log is True:
       P = dist.logpdf(X)
   return P

#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,nr_points,title="Title"):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters

    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""

	#npts = 100
    delta_x = float(xmax-xmin) / float(nr_points)
    delta_y = float(ymax-ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)


    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = multivariate_normal(mu, cov).pdf(pos)
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
    return
#--------------------------------------------------------------------------------
def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.

    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)

    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return np.random.permutation(y) # permutation of all samples
#--------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable.
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50]==0)   ,  np.sum(labels[0:50]==1)   , np.sum(labels[0:50]==2)   ],
                                  [np.sum(labels[50:100]==0) ,  np.sum(labels[50:100]==1) , np.sum(labels[50:100]==2) ],
                                  [np.sum(labels[100:150]==0),  np.sum(labels[100:150]==1), np.sum(labels[100:150]==2)]])
    new_labels = np.array([np.argmax(class_assignments[:,0]),
                           np.argmax(class_assignments[:,1]),
                           np.argmax(class_assignments[:,2])])
    return new_labels
#--------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)

    plot_gauss_contour(mu, cov, -2, 2, -2, 2,100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)

    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))

    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels =np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd==0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd==1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd==2] = new_labels[2]

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    #sanity_checks()
    main()