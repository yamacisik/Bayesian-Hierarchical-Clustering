=======
Final Project for Sta 663  by Matt Welch and Yamac Isik
=======


Bayesian Hierarchical Clustering
---------------------
An hiearchichal clustering method that uses a conjugate prior to form the clusters. The algorithm uses a tree node class to build the a tree resulting from the clustering process. 
For more information on the algorithm see : http://www2.stat.duke.edu/~kheller/bhcnew.pdf

Package is downloadable and installable via pip. 

- pip install bayescluster

or

- pip install --index-url https://test.pypi.org/simple/ bayescluster

Documentation
---------------------

**Cluster Function**

.. code-block:: python
>>> from bayes_cluster import cluster
>>> cluster.clust()
>>> clust(X,alpha,kappa0,v0,mu0,eta0,k)
>>>
  


 Input Variables:

- **X**: Features of the data set, must be a numpy array

- **alpha**:parameter to be set based on the number of clusters

- **kappa0, v0, mu0, eta0** :Parameters of the prior distribution, see the corresponding report for details.

- **k** : number of clusters to be determined

Returns **y, tree** 

 Output Variables:
- **y** :Numpy array giving the classifications of each observations

- **tree** : A tree node structure to be used in the impurity calculations


 **Impurity Measurement**
 
.. code-block:: python

  >>> from bayes_cluster import cluster,measures
  >>>measures.impurity(tree,yhat,y,n=500)
  >>>

- **tree**: tree node given by the algorithm

- **yhat**: predictions of the classifications

- **y**:original classification coming from the data set

- **n** :number of iterations







