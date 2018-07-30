"""
File: example.py
Author: Harry Sha, 2018
Description: An example demonstrating the usage of KMeans
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from custom_kmeans import KMeans
from scipy.spatial import distance

def custom_dist(u, v):
    """
    This is a custom distance function for the
    moons dataset. It assumes v is the local
    min/max of a parabola of the form 
    ax^2 + bx + c. Where a is eithet +1 or -1. 
    It the proposes a point on this parabola with 
    the same x coordinate as u. Then it takes 
    the min normal euclidean distance of v from 
    the proposed points. 

    This distance function is obviously super 
    contrived and only works well for this dataset
    but it illustrates what allowing for custom 
    distance functions can do. 

    """

    def quadratic(a, b, c, x):
        return a*x**2 + b*x + c

    a1, a2 = 1, -1
    b1, b2 = -2*v[0], 2*v[0]
    c1 = quadratic(-1*a1, -1*b1, v[1], v[0])
    c2 = quadratic(-1*a2, -1*b2, v[1], v[0])

    p1 = quadratic(a1, b1, c1, u[0]) - u[1]
    p2 = quadratic(a2, b2, c2, u[0]) - u[1]

    proposed = [p1**2, p2**2]
    return min(proposed)

def custom_av(points):
    """
    A custom average functions that pairs with the 
    custom distance function above. Used for the
    moons dataset.

    This custom average function finds the median 
    x coordinate and takes the corresponding 
    point as the average.
    """

    return points[points[:,0].argsort()[len(points)//2]]

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2)

    X = make_moons(200, noise=0.05)[0]

    km = KMeans(2, 5)
    labels = km.fit_predict(X)
    ax1.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
    ax1.set_title('Euclidean')

    km = KMeans(2, metric=custom_dist, average_fn=custom_av)
    labels = km.fit(X)
    ax2.scatter(X[:,0], X[:,1], c=km.labels_, cmap='viridis')
    ax2.set_title('Custom')

    print('labels:', km.labels_)
    print('cluster_centers:', km.cluster_centers_)
    print('inertia:', km.inertia_)
    plt.show()
    

