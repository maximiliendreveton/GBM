#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:50:36 2020

@author: mdreveto
"""


"""

# =============================================================================
# Figure 1: Eigenvector position 
# =============================================================================

N = 2000
rin = 0.08
rout = 0.02
accuracy_mean, accuracy_ste, good_eigenvalue_index = accuracy_as_function_eigenvector_order( N, rin, rout , number_of_eigenvectors = 100, naverage = 1, savefig = False, filename = "accuracy_function_index_eigenvector_N_?_rin_?_rout_?_naverage_?.eps", inverted = False)



# =============================================================================
# Figure 2: the first eigenvectors are indeed geometric
# =============================================================================

number_max_eigenvector_to_show = 10

N = 150
rin = 0.2
rout = 0.05
sizes = [ N//2, N//2 ]
G = gbm.GBM2d( sizes, rs = rin, rd = rout, tqdm_ = True )
labels_true = G.community_ground_truth

A = nx.adjacency_matrix( G )
vals, vecs = sp.sparse.linalg.eigsh( A.asfptype() , k = number_max_eigenvector_to_show, which = 'LA' )
#We need to reverse (flip) the order of eigenvalues, to have them from largest to smallest.
vals = np.flip( vals )
vecs = np.fliplr( vecs )

savefig = True
accuracy = [ ]
for eigenvector_index in tqdm( range( 1, number_max_eigenvector_to_show+1) ):
    labels_pred = [ ( vecs[ i, eigenvector_index - 1 ] > 0  ) * 1 + 1 for i in range( N ) ]
    accuracy.append( max( accuracy_score(labels_true, labels_pred), 1-accuracy_score(labels_true, labels_pred) ) )
    f = plt.figure()
    G.draw(node_color=labels_pred, node_size=25, alpha_node=0.8, alpha_edge= 0, width=0)
    filename = "eigenvector_" + str(eigenvector_index) + "_N_" + str(N) + "_rin_" + str(rin) + "_rout_" + str(rout) + ".eps"
    if(savefig):
        f.savefig(filename, format = 'eps')
    else:
        f.show()





# =============================================================================
# Figure 3 : Evolution of accuracy with N
# =============================================================================

rin = 0.08
rout = 0.05
#( mean_accuracies, ste_accuracies ) = plot_accuracy_with_n( rin, rout, N_range = [ 1000, 2500, 5000, 10000 ], methodsToCompare = [ 'HigherOrderSpectralClustering', 'HigherOrderSpectralClusteringWithLocalImprovement' ], n_average = 10, number_of_eigenvectors = 200, savefig = False, filename = "accuracy_varying_n_rin_?_rout_?_naverage_?.eps" )
( mean_accuracies, ste_accuracies ) = plot_accuracy_with_n( rin, rout, N_range = [ 500, 1000, 2000, 4000, 6000, 8000 ], methodsToCompare = [ 'HigherOrderSpectralClustering', 'HigherOrderSpectralClusteringWithLocalImprovement' ], n_average = 100, number_of_eigenvectors = 50, savefig = True, filename = "accuracy_varying_n_rin_0,08_rout_0,05_naverage_100.eps" )
#Takes about 5h on a laptop
#Note: actually could easily increase speed, beacuse HigherOrderSpectralClusteringWithLocalImprovement could re-use the label predicton of HigherOrderSpectralClustering, but I'm lazy to implement it.



# =============================================================================
# Figure 5: Comparison of HOSC with triangle countings algorithms
# =============================================================================

    
N = 3000
rout = 0.04
algosToCompare = ['HOSC', 'Motif Counting 1', 'Motif Counting 2']
rin_range = np.linspace( rout, 0.15, num= 12 )
n_average = 50
(mean_accuracies, ste_accuracies) = compare_clustering_methods ( N, rout, rin_range = rin_range, methodsToCompare = algosToCompare, number_of_tries = n_average, number_of_eigenvectors = 100 , savefig=False, filename = 'comparision_methods_N_3000_rout_0.04_naverage_50_step_0,01.eps')
#Took 35h on laptop; redo and go to rin = 0.2 ? (at least try and see what happens, when does Motif Counting 1 will finally work ?)


#TODO:
N = 2000    
rout = 0.1
rin_range = np.linspace( 0.0, 0.35, num= 36 )
algosToCompare = ['HOSC', 'Motif Counting 1', 'Motif Counting 2']
n_average = 1 #1h10 per average on laptop
(mean_accuracies, ste_accuracies) = compare_clustering_methods ( N, rout, rin_range = rin_range, methodsToCompare = algosToCompare, number_of_tries = n_average, savefig=False, filename = 'gbm_comparision_methods_N_2000_rout_0.1_naverage_?.eps')
#On inria: about 10min per data point per average (depends how sparse it is)
#Good but have to go to rin = 0.3 so that Motif Counting 2 works (quite dense case).


#TODO
N = 5000    
rout = 0.04
algosToCompare = ['HOSC', 'Motif Counting 1', 'Motif Counting 2']
rin_range = np.linspace( rout, 0.15, num= 12 )
n_average = 1
(mean_accuracies, ste_accuracies) = compare_clustering_methods ( N, rout, rin_range = rin_range, methodsToCompare = algosToCompare, number_of_tries = n_average, number_of_eigenvectors = 50 , savefig=True, filename = 'comparision_methods_N_5000_rout_0.04_naverage_1.eps')
#3h for 1 average on laptop. Curve looks good though.
#rin < 0.04 doesn't work, no need to try.


# =============================================================================
# Figure 4: Dip in accuracy for a finite number of rin  (rout fixed)
# =============================================================================

N = 3000
rout = 0.06
n_average = 5  
#1h per average on laptop
mean_accuracies, ste_accuracies, indexes_ideal_eigenvalues = dip_explanation( N, rout, rin_range = np.linspace( 0.08, 0.2, num= 161 ), n_average = n_average, savefig = True, filename = 'gbm_accuracy_dip_N_3000_rout_0.06_naverage_5_npoints_161_step_0,00075.eps' )


N = 3000
rout = 0.04
n_average = 1
#30min per average on Inria computer
mean_accuracies, ste_accuracies, indexes_ideal_eigenvalues = dip_explanation( N, rout, rin_range = np.linspace( 0.06, 0.2, num= 201 ), n_average = n_average, savefig = True, filename = 'gbm_accuracy_dip_N_3000_rout_0.04_naverage_1_npoints_201_step_0,0007.eps' )




"""

import numpy as np 
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tqdm import tqdm


import class_gbm as gbm
import SpectralClustering as sc
import triangle_counting_umass_implementation as umass_implementation
import triangle_counting_personal_implementation as counting_triangles_algo
import higher_order_spectral_clustering as hosc





SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18




# =============================================================================
# Clustering algorithms
# =============================================================================

def clusteringGBM(G, method = 'HigherOrderSpectralClustering', number_of_eigenvectors = 20):
    """
    This function takes a Graph and a method (plus the number_of_eigenvectors to compute is needed),
    and output the accuracy given by the method
        The number_of_eigenvectors is used for HOSC algo, only if the graph is marked as sparse
        If the graph is not sparse, then it will compute all eigenvectors (using numpy linalg)
        
    The parameters a, b, labels_true, number_vectors might be needed for some methods, and not needed for others.
    """
    
    if (method == 'HigherOrderSpectralClustering' or method == 'HOSC'):
        rin = G.intra_cluster_threshold
        rout = G.inter_cluster_threshold
        return hosc.higherOrderSpectralClustering(G, 2 * rin, 2 * rout, number_of_eigenvectors = number_of_eigenvectors, sparse = False )
    
    if (method == 'HigherOrderSpectralClusteringWithLocalImprovement' or method == 'HOSC-LI'):
        rin = G.intra_cluster_threshold
        rout = G.inter_cluster_threshold
        labelsPred_after_first_step = hosc.higherOrderSpectralClustering(G, 2 * rin, 2 * rout, number_of_eigenvectors = number_of_eigenvectors, sparse = ( (rin + rout) / 2 < 0.4 ) )
        return hosc.localImprovement( G, labelsPred_after_first_step )


    if (method == 'SpectralClustering'):
        return sc.spectralClustering(G, K = 2, method = "normalized" )
    
    elif (method == 'Umass_second_algo_personal_implementation'):
        rs = G.intra_cluster_threshold
        rd = G.inter_cluster_threshold
        return counting_triangles_algo.countingUmassLastPaper(G, rs, rd)
    
    elif(method == 'Umass_first_algo_personal_implementation'):
        #Somehow this implementation work worse that the code they send us
        #But this implementation is supposed to be a perfect replica of the algorithm described in their first paper
        #while their implementation is something weird.
        rs = G.intra_cluster_threshold
        rd = G.inter_cluster_threshold
        return counting_triangles_algo.simpleTriangleCounting(G, rs, rd)
        
    elif (method == 'Umass_first_algo' or method == 'Motif Counting 1'):
        rs = G.intra_cluster_threshold
        rd = G.inter_cluster_threshold
        return umass_implementation.Umass_old_algo(G, rs, rd)
    
    elif (method == 'Umass_second_algo' or method == 'Motif Counting 2'):
        rs = G.intra_cluster_threshold
        rd = G.inter_cluster_threshold
        return umass_implementation.Umass_second_algo(G, rs, rd)
    
    return 0





# =============================================================================
# Plotting functions
# =============================================================================


def plot_accuracy_with_n( rin, rout, N_range = [ 1000, 2500, 5000, 10000 ], methodsToCompare = ['HigherOrderSpectralClustering'] , n_average = 10, number_of_eigenvectors = 200, savefig = False, filename = "accuracy_varying_n_rin_?_rout_?_naverage_?.eps" ):
    
    mean_accuracies = dict( )
    ste_accuracies = dict( )
    for method in methodsToCompare:
        mean_accuracies[method] = []
        ste_accuracies[method] = []


    for n in tqdm( N_range ):
        sizes = [ n//2, n//2 ]
        accuracies = dict()                
        
        for method in methodsToCompare:
            accuracies[method] = []

        for trial in range( n_average ):
            G = gbm.GBM2d( sizes, rin, rout )
            labels_true = G.community_ground_truth
            
            for method in methodsToCompare:
                labels_pred = clusteringGBM( G, method = method, number_of_eigenvectors = number_of_eigenvectors )
                accuracies[method].append( max(accuracy_score(labels_true, labels_pred) , 1-accuracy_score(labels_true, labels_pred)  ) )


        for method in methodsToCompare:
            #print('method : ', method, 'accuracy : ', accuracies[method] )
            mean_accuracies[method].append( np.mean( accuracies[method] ) )
            ste_accuracies[ method ].append( np.std( accuracies[method] ) / np.sqrt( n_average ) )
            
    for method in methodsToCompare:
        if method == 'HigherOrderSpectralClustering':
            label = 'HOSC'
        elif method == 'HigherOrderSpectralClusteringWithLocalImprovement':
            label = 'HOSC-LI'
        else:
            label = method
        plt.errorbar( N_range , mean_accuracies[ method ], yerr = ste_accuracies[ method ], linestyle = '-.', label= label )
    legend = plt.legend( title="Algorithm:", loc=0,  fancybox=True, fontsize = SIZE_LEGEND )
    plt.setp( legend.get_title(), fontsize= SIZE_LEGEND )
    plt.xlabel("n", fontsize = SIZE_LABELS)
    plt.ylabel("Accuracy", fontsize = SIZE_LABELS)
    plt.xticks( [2000, 4000, 6000, 8000], fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    plt.title("Evolution of accuracy with n, \n for rin = %s and rout = %s " % (rin, rout) , fontsize = SIZE_TITLE )
    if (savefig):
        plt.savefig(filename, format = 'eps', bbox_inches='tight' )

    plt.show()
    
    return ( mean_accuracies, ste_accuracies )



def accuracy_as_function_eigenvector_order( N, rin, rout , number_of_eigenvectors = 50, naverage = 1, savefig = True, filename = "accuracy_function_index_eigenvector_N_?_rin_?_rout_?_naverage_?.eps", inverted = False):
    
    accuracy_mean = []
    accuracy_ste = []
    
    accuracy = np.zeros( ( naverage, number_of_eigenvectors ) )
    
    sizes = [ N//2, N//2 ]
    good_eigenvalue_index = [ ]
    
    for trial in tqdm( range (naverage ) ):
        
        G = gbm.GBM2d( sizes, rs = rin, rd = rout )
        labels_true = G.community_ground_truth
        A = nx.adjacency_matrix( G )
         
        if(inverted):
            vals, vecs = sp.sparse.linalg.eigsh( A.asfptype() , k= number_of_eigenvectors, which = 'SA' )
        else:
            vals, vecs = sp.sparse.linalg.eigsh( A.asfptype() , k= number_of_eigenvectors, which = 'LA' )
            #We need to reverse (flip) the order of eigenvalues, to have them from largest to smallest.
            vals = np.flip( vals )
            vecs = np.fliplr( vecs )
        
        #vals, vecs = np.linalg.eigh( - A.todense() )

        for k in range( number_of_eigenvectors ):
            labels_pred = [ ( vecs[i, k] > 0  ) *1 + 1 for i in range( N ) ]
            accuracy[trial, k] = max( accuracy_score(labels_pred, labels_true), 1-accuracy_score(labels_pred, labels_true)  )
        
        theoretical_value_of_the_good_eigenvalue = N * ( rin - rout )
        
        distances = [np.abs( val - theoretical_value_of_the_good_eigenvalue ) for val in vals ]
        good_eigenvalue_index.append( np.argmin( distances ) )
    
    for k in range( number_of_eigenvectors ):
        accuracy_mean.append (np.mean( accuracy[ : , k ] ) )
        accuracy_ste.append( np.std( accuracy[ :,k ] / np.sqrt( naverage ) ) )
        
    if naverage == 1:
        fig = plt.errorbar( range(number_of_eigenvectors), accuracy_mean, yerr = accuracy_ste, linestyle = '', marker = 'o' )
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot( range(number_of_eigenvectors), accuracy_mean )
        #ax.set_aspect(aspect= 'auto' )
        
    else:
        plt.errorbar( range(number_of_eigenvectors), accuracy_mean, yerr = accuracy_ste, linestyle = '-' )
    #plt.legend(title="Comparison of Algorithms:", loc=0,  fancybox=True)
    plt.xlabel( "Eigenvector index", fontsize = SIZE_LABELS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
#    plt.title("Evolution of accuracy with the index of eigenvector, \n for N = %s, rin = %s and rout = %s " % (N, rin, rout) , fontsize = SIZE_TITLE )
    if (savefig):
        plt.savefig(filename, format = 'eps', bbox_inches='tight' )

    plt.show()
        
    return accuracy_mean, accuracy_ste, good_eigenvalue_index




def compare_clustering_methods (N, rout, rin_range = np.linspace(0.01, 0.3, num=5), methodsToCompare = ['SpectralClustering', 'Umass_first_algo'],
                         number_of_tries = 5, number_of_eigenvectors = 20, 
                         plotFig = True, savefig = False, filename = "comparision_methods_N_?_rout_?_naverage_?.eps", xtick = 0.04, legend = True):
    
    sizes = [ N//2, N//2 ]
    
    mean_accuracies = dict( )
    ste_accuracies = dict( )
    for method in methodsToCompare:
        mean_accuracies[method] = []
        ste_accuracies[method] = []
        
    for i in tqdm( range( len( rin_range ) ) ):        
        accuracies = dict()                
        
        for method in methodsToCompare:
            accuracies[method] = []
            
        for trial in range(number_of_tries):
            G = gbm.GBM2d( sizes, rs = rin_range[ i ], rd = rout )
            labels_true = G.community_ground_truth
            
            for method in methodsToCompare:
                labels_pred = clusteringGBM( G, method = method, number_of_eigenvectors = number_of_eigenvectors )
                accuracies[method].append( agreement(labels_pred, labels_true)  )
                    
        for method in methodsToCompare:
            #print('method : ', method, 'accuracy : ', accuracies[method] )
            mean_accuracies[method].append( np.mean( accuracies[method] ) )
            ste_accuracies[ method ].append( np.std( accuracies[method] ) / np.sqrt( number_of_tries ) )

    if(plotFig):
        for method in methodsToCompare:
            plt.errorbar( rin_range, mean_accuracies[ method ], yerr = ste_accuracies[ method ], linestyle = '-.', label= method )
        if legend:
            legend = plt.legend( title="Algorithm:", loc=0,  fancybox=True )
            plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )
        plt.xlabel( "rin", fontsize = SIZE_LABELS )
        plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
        plt.xticks( np.arange(min( rin_range ), max( rin_range ) + 0.01, xtick ), fontsize = SIZE_TICKS )
        plt.yticks( fontsize = SIZE_TICKS )
        plt.title("Evolution of accuracy, \n for n = %s and rout = %s " % ( N, rout ), fontsize = SIZE_TITLE )
        if (savefig):
            plt.savefig(filename, format = 'eps', bbox_inches='tight' )

    return mean_accuracies, ste_accuracies





# =============================================================================
# Eigenvalue study
# =============================================================================

def gbm_kth_fourier_mode( r, k ):
    if  r == 0:
        return 0
    elif k == 0:
        return 2 * r
    else:
        return 2 * r * np.sin ( 2 * np.pi * k * r) / (2 * np.pi * k * r)


def gbm_fourier_mode( r, N ):
    return [ gbm_kth_fourier_mode( r, k )  for k in np.arange( -N//2, N//2, 1 ) ]


def gbm_position_ideal_eigenvalue( N, rin, rout ):
    fourier_mode_in = gbm_fourier_mode(rin, N//2)
    fourier_mode_out = gbm_fourier_mode(rout, N//2)
    
    eigenvalue = [ fourier_mode_in[k] + fourier_mode_out[k] for k in range(len(fourier_mode_in)) ] + [ fourier_mode_in[k] - fourier_mode_out[k] for k in range(len(fourier_mode_in)) ]
    
    eigenvalue_sorted = np.sort(eigenvalue)
    eigenvalue_sorted_decreasing = np.flip( eigenvalue_sorted )
    
    ideal_eigenvalue = gbm_kth_fourier_mode(rin, 0) - gbm_kth_fourier_mode(rout, 0)
    
    return list( eigenvalue_sorted_decreasing ).index(ideal_eigenvalue)
    
    

# =============================================================================
# Explanation of the dips
# =============================================================================



def evolution_index_ideal_eigenvalue( N, rout, rin_range = np.linspace(0, 0.2, num = 21) ):
    indexes_ideal_eigenvalues = []
    
    for rin in rin_range:
        indexes_ideal_eigenvalues.append( gbm_position_ideal_eigenvalue( N, rin, rout ) )
    
    #plt.plot(rin_range, indexes_ideal_eigenvalues, linestyle = '', marker = 'o')
    
    
    return indexes_ideal_eigenvalues
    
    
def dip_explanation( N, rout, rin_range = np.linspace( 0.08, 0.2, num= 161 ), n_average = 1, savefig=False, filename = 'comparision_methods_N_3000_rout_0.06_naverage_?_npoints_161_step_?.eps' ):
    
    (mean_accuracies, ste_accuracies) = compare_clustering_methods ( N, rout, rin_range = rin_range, methodsToCompare = ['HOSC'], number_of_tries = n_average, plotFig = False )
    indexes_ideal_eigenvalues = evolution_index_ideal_eigenvalue( N, rout, rin_range = rin_range )
       
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel( 'rin', fontsize = SIZE_LABELS )
    ax1.tick_params( axis = 'x', labelsize = SIZE_TICKS)
    ax1.set_ylabel( 'Accuracy', color=color, fontsize = SIZE_LABELS )
    ax1.plot( rin_range, mean_accuracies['HOSC'], color=color )
    ax1.tick_params( axis='y', labelcolor=color, labelsize = SIZE_TICKS )
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:red'
    ax2.set_ylabel( 'Ideal eigenvector index', color=color, fontsize = SIZE_LABELS - 2)  # we already handled the x-label with ax1
    ax2.plot( rin_range, indexes_ideal_eigenvalues, color=color, linestyle = '', marker = '.' )
    ax2.set_yticks ( [ 3, 5, 7, 9, 11, 13] )
    ax2.tick_params( axis='y', labelcolor=color, labelsize = SIZE_TICKS )
    
    fig.tight_layout( )  # otherwise the right y-label is slightly clipped
    if (savefig):
            plt.savefig(filename, format = 'eps', bbox_inches='tight' )

    plt.show( )
    
    
    return mean_accuracies, ste_accuracies, indexes_ideal_eigenvalues


# =============================================================================
# Miscellaneous
# =============================================================================

def agreement(labels_pred, labels_true):
    """
    This function computes the accuracy when there should be 2 communities of same size,
    but the labels_pred vector might induce several predicted communities
    This is necessary because in some situation, Umass algorithms predict more than 2 communities
    
    Example:
        labels_true uses labels 1, 2
        labels_pred uses labels 0,1
        
        or :
        labels_true uses labels 1, 2
        labels_pred uses labels 1, 2, 3
    """
    true_communities_labels = set( labels_true )
    predicted_communities_labels = set( labels_pred )
    if( true_communities_labels == predicted_communities_labels ):
        return max( accuracy_score(labels_true, labels_pred) , 1 - accuracy_score(labels_true, labels_pred) )
    elif len( predicted_communities_labels ) == 1:
        return max( accuracy_score(labels_true, labels_pred) , 1 - accuracy_score(labels_true, labels_pred) )
    else:
        N = len( labels_pred )
        predicted_communities_labels = list( predicted_communities_labels )
        community_size = [ ]
        for label in predicted_communities_labels:
            community_size.append( len( [ i for i in range( N ) if labels_pred[ i ] == label ] ) )
            
        largest_community_labels = [ predicted_communities_labels[ np.argsort(community_size)[-k-1] ] for k in range( len(true_communities_labels) ) ]
        
        
        if (-250 not in true_communities_labels):
            new_labels_pred = np.ones( N ) * (-250)
            true_communities_labels = list( true_communities_labels )
            good_nodes = []
            for i in range(N):
                if labels_pred[i] in largest_community_labels:
                    new_labels_pred[ i ] = true_communities_labels[ largest_community_labels.index( labels_pred[i] )  ]
                    good_nodes.append( i )
            count = 0
            for i in good_nodes:
                if new_labels_pred[i] == labels_true[i]:
                    count += 1
            return max( 0.5, 1/N * max(count, len(good_nodes)-count) )
    
    return 0
