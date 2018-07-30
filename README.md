# custom_kmeans
Implementation of the k-means algorithm with customizable distance, and averaging function

## Motivation
The k-means algorithm is a well-known unsupervised learning method that can identify clusters in data. The original version of the k-means algorithm uses Euclidean distance between points as a measure of similiarity/difference. However due to the [*the curse of dimensionality*](https://en.wikipedia.org/wiki/Curse_of_dimensionality#Distance_functions), Euclidean distance may not be the best option. Therefore, we may want to try different distance functions. A commonly proposed distance function is to use cosine distance. As of July 2018, the sklearn implementation of KMeans does not support changing distance metrics and averaging functions, so I've implemented a version here.

Aside from dimensionality, another problem that is tackled by customizable distance and averaging functions is 'weirdly' shaped clusters.

## Usage
Requires numpy and scipy.

This implementation is modelled after [scikit learn's implemenation](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.transform), meaning that it has many of the same function/variable names.

To use, import like this: `from custom_kmeans import KMeans`.

## Docs
class KMeans(k, n_init=10, metric=distance.euclidean, 
        average_fn=lambda x: np.mean(x, axis=0), max_iter=100)
###Parameters
k <int>: Number of clusters

n_init <int>: Number of initializations to try

metric <fn>: Function that takes in two arrays and returns a number
                (used as the distance metric)

average_fn <fn>: A function that takes in a matrix X
                    and returns the center of the points.

max_iter <int>: Maximum number of iterations to run before the 
                    algorithm terminates

###Attributes
inertia_ <float>: Sum of distances for each point to its centroid
                    Where distance is taken as the metric argument.

cluster_centers_ <2d array>: Each row represents the coordinates of one centroid

labels_ <array>: A list of assignments.

###Functions
fit, fit_predict, predict

These behave as in [scikit learn's implemenation](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.transform)






