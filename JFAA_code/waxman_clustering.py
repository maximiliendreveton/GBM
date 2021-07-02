#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:54:56 2020

@author: mdreveto
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tqdm import tqdm

import class_sgbm as sgbm
import higher_order_spectral_clustering as hosc


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 16



"""


# =============================================================================
# Figures when sout varies
# =============================================================================
parameters = dict( )
parameters["qin"] = 0.5
parameters["qout"] = 0.5
parameters["sin"] = 1.0

accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter (fixedParameters = parameters, varyingParameters = np.linspace(0,1,11), N = 1000, whatVaries = "sout", n_average = 10, savefig = True, filename = "Waxman_model_N_1000_qin_0,5_qout_0,5_sin_1,0_naverage10.eps" )


# =============================================================================
# Figures when qout varies
# =============================================================================
parameters = dict( )
parameters["qin"] = 0.4
parameters["sin"] = 0.2
parameters["sout"] = 0.2


accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter (fixedParameters = parameters, varyingParameters = np.linspace(0,1,11), N = 1000, whatVaries = "qout", n_average = 10, savefig = True, filename = "Waxman_model_N_1000_qin_0,6_sin_0,2_sout_0,2_naverage10.eps" )

accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter (fixedParameters = parameters, varyingParameters = np.linspace(0,1,51), N = 1000, whatVaries = "qout", n_average = 10, savefig = True, filename = "Waxman_model_N_1000_qin_0,6_sin_0,2_sout_0,2_naverage10_linespace_51.eps")



# =============================================================================
# Figures when n varies
# =============================================================================

parameters = dict( )
parameters["qin"] = 0.5
parameters["qout"] = 0.5
parameters["sin"] = 2.0
n_average = 10
accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter_different_n (fixedParameters = parameters, varyingParameters = np.linspace(0, 5.0, 25), N_range = [500, 1000, 2000, 4000], whatVaries = "sout", model = 'Waxman', n_average = n_average, savefig = True, filename = "Waxman_model_different_N_qin_0,5_qout_0,5_sin_2,0_naverage_10.eps", xtick=1.0 )


parameters = dict( )
parameters["qin"] = 0.4
parameters["sin"] = 1.0
parameters["sout"] = 1.0
n_average = 10
accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter_different_n (fixedParameters = parameters, varyingParameters = np.linspace(0, 0.6, 21), N_range = [500,1000, 2000, 4000], whatVaries = "qout", model = 'Waxman', n_average = n_average, savefig = True, filename = "Waxman_model_different_N_qin_0,4_sin_1,0_sout_1,0_naverage_10.eps" )




# =============================================================================
# Fin, Fout Gamma function
# =============================================================================
parameters = dict( )
parameters["qin"] = 1.0
parameters["sin"] = 3.0
parameters["sout"] = 1.0
accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter_different_n (fixedParameters = parameters, varyingParameters = np.linspace(0, 0.6, 21), N_range = [500,1000, 2000, 4000], whatVaries = "qout", model = 'Gamma', n_average = 4, savefig = True, filename = "Waxman_model_different_N_qin_0,4_sin_1,0_sout_1,0_naverage_4.eps" )





#TODO on Inria computer: simulations with more points (in linspace)
#Each simulation takes 20-25min times n_average .

parameters = dict( )
parameters["qin"] = 10
parameters["sin"] = 10
parameters["sout"] = 10
n_average = 10
accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter_different_n (fixedParameters = parameters, varyingParameters = np.linspace(6, 16, 21), N_range = [500,1000, 2000, 4000], whatVaries = "qout", model = 'Gamma', n_average = n_average, savefig = True, filename = "Gamma_model_different_N_qin_010_sin_10_sout_10_naverage_10.eps" , xtick=2)

parameters = dict( )
parameters["qin"] = 7
parameters["sin"] = 5
parameters["sout"] = 5
accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter_different_n (fixedParameters = parameters, varyingParameters = np.linspace(3, 9, 21), N_range = [ 500, 1000, 2000, 4000 ], whatVaries = "qout", model = 'Gamma', n_average = 4, savefig = True, filename = "Gamma_model_different_N_qin_7_sin_5_sout_5_naverage_4.eps", xtick=2 )


parameters = dict( )
parameters["qin"] = 7
parameters["sin"] = 10
parameters["sout"] = 10
accuracy_mean, accuracy_ste = plotAccuracyVaryingOneParameter_different_n (fixedParameters = parameters, varyingParameters = np.linspace(3, 11, 21), N_range = [500,1000, 2000, 4000], whatVaries = "qout", model = 'Gamma', n_average = 4, savefig = True, filename = "Gamma_model_different_N_qin_7_sin_10_sout_10_naverage_4.eps" , xtick=2)




"""


def negativeExponential( x, q, s ):
    return min(1, q * np.exp(- s * x) )


def gamma( x, q, s ):
    return min(1, q * x * np.exp(- s * x) )



def firstFourierModeNegativeExponential( q, s ):
    if s==0:
        return q
    else:
        return 2*q / s * ( 1 - np.exp(-s/2) )
    
    
def firstFourierModeGamma( q, s ):
    #TOD; verify computations
    if s==0:
        return q
    else:
        return 2*q / s * (  - np.exp(-s/2) / 2 + 1/s * ( 1 - np.exp(-s/2) ) )



def plotAccuracyVaryingOneParameter( fixedParameters, varyingParameters, N = 1000, whatVaries = "sout", model = "Waxman", n_average = 10, plotFig = True, savefig = False, filename = "Waxman_model_naverage?.esp" , makeTitle = True, tqdm_ = True ):
    accuracy_mean = []
    accuracy_ste = []

    sizes = [ N//2, N//2 ]

    if tqdm_:
        loop = tqdm( varyingParameters )
    else:
        loop = varyingParameters
    
    for parameter in loop:
        fixedParameters[ whatVaries ] = parameter
        qin = fixedParameters["qin"]
        qout = fixedParameters["qout"]
        sin = fixedParameters["sin"]
        sout = fixedParameters["sout"]

        if model == 'Waxman':
            fin = lambda x : negativeExponential( x, qin, sin )
            fout = lambda x : negativeExponential( x, qout, sout ) 
            muin = firstFourierModeNegativeExponential( qin, sin )
            muout = firstFourierModeNegativeExponential( qout, sout )
        elif model == 'Gamma':
            fin = lambda x : qin * x * np.exp(- sin * x)
            fout = lambda x : qout * x * np.exp(- sout * x)
            muin = firstFourierModeGamma( qin, sin )
            muout = firstFourierModeGamma( qout, sout )

        
        accuracies = [ ]
        for trial in range( n_average ):
            G = sgbm.StochasticGeometricBlockModel( sizes, fin, fout )
            labels_true = G.community_ground_truth
            
            labels_pred = hosc.higherOrderSpectralClustering( G, muin, muout, sparse = False )
            accuracies.append( max(accuracy_score(labels_true, labels_pred) , 1-accuracy_score(labels_true, labels_pred)  ) )

        accuracy_mean.append( np.mean( accuracies ) )
        accuracy_ste.append( np.std( accuracies ) / np.sqrt( n_average ) )

    if( plotFig ):
        plt.errorbar( varyingParameters, accuracy_mean, yerr = accuracy_ste, linestyle = '-.')
        plt.xlabel( whatVaries, fontsize = SIZE_LABELS )
        plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
        plt.xticks( fontsize = SIZE_TICKS )
        plt.yticks( fontsize = SIZE_TICKS )
    
        plt.xticks( np.arange(0, max( varyingParameters ) + 0.01, 0.5), fontsize = SIZE_TICKS )
        
        if (makeTitle == True):
            if whatVaries == "sout":
                plt.title( "Evolution of accuracy with sout. \n N = %s, qin = %s, qout = %s, sin = %s" % (N, qin, qout, sin) , fontsize = SIZE_TITLE )
            elif whatVaries == "sin":
                plt.title( "Evolution of accuracy with sin. \n N = %s, qin = %s, qout = %s, sout = %s" % (N, qin, qout, sout) , fontsize = SIZE_TITLE )
            elif whatVaries == "qin":
                plt.title( "Evolution of accuracy with qin. \n N = %s, qout = %s, sin = %s, sout = %s" % (N, qout, sin, sout) , fontsize = SIZE_TITLE )
            elif whatVaries == "qout":
                plt.title( "Evolution of accuracy with qout. \n N = %s, qin = %s, sin = %s, sout = %s" % (N, qin, sin, sout) , fontsize = SIZE_TITLE )
    
        if (savefig):
            plt.savefig(filename, format = 'eps', bbox_inches='tight' )
    
        plt.show()
    
    return accuracy_mean, accuracy_ste
    


def plotAccuracyVaryingOneParameter_different_n (fixedParameters, varyingParameters, N_range = [1000, 2000], whatVaries = "sout", model = "Waxman", n_average = 10, savefig = False, filename = "Waxman_model_naverage?.esp" , makeTitle = True, xtick = 0.2, legend = True):
    accuracy_mean = []
    accuracy_ste = []

    for n in tqdm( N_range ):
        ( accuracy_mean_n, accuracy_ste_n ) = plotAccuracyVaryingOneParameter( fixedParameters, varyingParameters, N = n, whatVaries = whatVaries, model = model, n_average = n_average, plotFig = False, tqdm_ = False)
        accuracy_mean.append( accuracy_mean_n )
        accuracy_ste.append( accuracy_ste_n )
    
    for k in range( len(N_range) ):
        plt.errorbar( varyingParameters, accuracy_mean[ k ], yerr = accuracy_ste[ k ], linestyle = '-.', label= "n = " + str(N_range[k]) )
    if(legend):
        legend = plt.legend(title="Graph size:", loc=0,  fancybox=True, fontsize = SIZE_LEGEND)
        plt.setp( legend.get_title(), fontsize= SIZE_LEGEND )
    plt.xlabel( whatVaries, fontsize = SIZE_LABELS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    plt.xticks( np.arange(min(varyingParameters), max( varyingParameters ) + 0.01, xtick ), fontsize = SIZE_TICKS )
    
    if (makeTitle == True):
        if whatVaries == "sout":
            qin = fixedParameters["qin"]
            qout = fixedParameters["qout"]
            sin = fixedParameters["sin"]
            plt.title( "Evolution of accuracy with sout. \n qin = %s, qout = %s, sin = %s" % (qin, qout, sin) , fontsize = SIZE_TITLE )

        elif whatVaries == "qout":
            qin = fixedParameters["qin"]
            sout = fixedParameters["sout"]
            sin = fixedParameters["sin"]
            plt.title( "Evolution of accuracy with qout. \n qin = %s, sin = %s, sout = %s" % (qin, sin, sout) , fontsize = SIZE_TITLE )

    if (savefig):
        plt.savefig(filename, format = 'eps', bbox_inches='tight' )

    plt.show()
    
    return accuracy_mean, accuracy_ste



def plotGammaFunction( x_range = np.linspace(-0.5, 0.5, 101), s_range = [0, 1, 2, 5, 10], q=1, xtick = 0.25, savefig = True, filename='gamma_function_example.eps' ) :
    y = []
    
    for s in s_range:
        y.append( [ gamma(np.abs(x), q, s ) for x in x_range ] )
    
    for k in range(len(s_range) ):
        plt.plot( x_range, y[k], label = 's = ' + str(s_range[k]) )
    legend = plt.legend(title="Parameter s:", loc=0,  fancybox=True, fontsize = SIZE_LEGEND)
    plt.setp( legend.get_title(), fontsize= SIZE_LEGEND )
    plt.xlabel( 'x', fontsize = SIZE_LABELS )
    #plt.ylabel( "f(x) = x exp(-s x)", fontsize = SIZE_LABELS )
    plt.xticks( fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    plt.xticks( np.arange(min(x_range), max( x_range ) + 0.01, xtick ), fontsize = SIZE_TICKS )
    plt.title( "Graph of F(x) = x exp(-s x)" , fontsize = SIZE_TITLE )
    
    if(savefig):
        plt.savefig(filename, format = 'eps', bbox_inches='tight' )
    
    return 0