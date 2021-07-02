#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:03:23 2020

@author: mdreveto
"""

import numpy as np 
import networkx as nx
import scipy as sp



def higherOrderSpectralClustering( G, muin, muout, sparse = False, number_of_eigenvectors = 10 ):
    
    N = nx.number_of_nodes( G )
    labels_pred = np.zeros( N, dtype = int )
    
    if(not isinstance(number_of_eigenvectors, int) or number_of_eigenvectors == 0):
        return print("Choose an appropriate number of eigenvectors")
    
    A = nx.adjacency_matrix( G )
    
    inverted = ( muin < muout )
    
    if ( sparse and number_of_eigenvectors < N/2 ): 
        #If graph is not sparse or number_of_eigenvectors too large, it is inefficient to use methodss from scipy sparse library
        if(inverted):
            vals, vecs = sp.sparse.linalg.eigsh( A.asfptype() , k=number_of_eigenvectors, which = 'SA' )
        else:
            vals, vecs = sp.sparse.linalg.eigsh( A.asfptype() , k=number_of_eigenvectors, which = 'LA' )
            #We need to reverse (flip) the order of eigenvalues, to have them from largest to smallest.
            vals = np.flip( vals )
            vecs = np.fliplr( vecs )

    else:
        vals, vecs = np.linalg.eigh( A.todense() )
        vals = np.flip( vals )
        vecs = np.fliplr( vecs )
        vecs = np.asarray(vecs)

    
    expected_value_of_the_ideal_eigenvalue = ( muin - muout ) * N / 2

    distances = [np.abs( val - expected_value_of_the_ideal_eigenvalue ) for val in vals ]
    good_eigenvalue_index = np.argmin( distances )
    if sparse and good_eigenvalue_index == number_of_eigenvectors-1:
        print("You might want to increase the number_of_eigenvectors")
    
    #print ('Le bon eigenvector est en position : ', good_eigenvalue_index)
    #print ('La distance est de ', np.min( distances) )
    
    for i in range( N ):
        labels_pred[i] = ( vecs[ i, good_eigenvalue_index ] > np.median( vecs[ :, good_eigenvalue_index ] )  ) * 1 + 1
    
    return labels_pred




def localImprovement( G, labelsPredicted ):
    N = nx.number_of_nodes( G )
    labelsPredictedUpdated = np.zeros( N, dtype = int )
    
    A = nx.adjacency_matrix( G )
    Z = np.zeros( ( N,2 ) )
    for i in range( N ):
        Z[ i, int( labelsPredicted[ i ] ) -1 ] = 1
    
    neighbors = A @ Z
    
    for i in range( N ):
        labelsPredictedUpdated[ i ] = np.argmax( neighbors[ i, : ] ) + 1
    
    return labelsPredictedUpdated
