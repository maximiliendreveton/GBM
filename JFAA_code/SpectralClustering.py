#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:45:23 2020

@author: mdreveto
"""

import networkx as nx
import scipy as sp
import numpy as np
from sklearn.cluster import KMeans



def spectralClustering(G, K = 2, method = "normalized"):
    #detect communities from the sign of the second eigenvector of the Laplacian matrix
    if not isinstance(K, int):
        raise TypeError("The number of communities K should be an integer")
    if K<2:
        raise TypeError("The number of communities K should be greater or equal that 2")
    if not isinstance(G, nx.Graph):
        raise TypeError("The graph G should be an instance of the class Graph of networkx package")
    
    if (method == "standard"):
        matrix = nx.laplacian_matrix(G)
    elif (method == "normalized"):
        matrix = nx.normalized_laplacian_matrix(G)
    elif (method == "random_walk"):
        A = nx.adjacency_matrix(G)
        D = sum(A)
        Dinvert = np.diag( [ 1/D[0,count] for count in range( nx.number_of_nodes(G) ) ] )
        matrix = Dinvert.dot( nx.laplacian_matrix(G).todense() )
        matrix = sp.sparse.csr_matrix(matrix)
    elif(method == "adjacency_matrix"):
        matrix = - nx.adjacency_matrix(G)
    else:
        raise TypeError("The method is not implemented")
        
    vals, vecs = sp.sparse.linalg.eigsh(matrix.asfptype() , k= K, which = 'SM')
    
    kmeans = KMeans( n_clusters = K, random_state=0 ).fit( vecs )
    labels_pred = kmeans.labels_ + np.ones( nx.number_of_nodes(G) )
    return labels_pred.astype(int)



def spectralClusteringRegularization(G, K = 2):
    n = nx.number_of_nodes(G)
    A = nx.adjacency_matrix(G).todense() + nx.number_of_edges(G) / (2*n) * np.ones( (n,n) )
    G_regularized = nx.from_numpy_array(A)
    
    return spectralClustering(G_regularized)
