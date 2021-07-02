#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 7 10:30:30 2019

@author: mdreveto
"""

import networkx as nx
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

import random as random


################ UMASS guys New algo
"""
n=1000
a = 70
b = 7
G = gg.GBM2d( n, a*np.log(n)/n, b*np.log(n)/n )

labels_pred = countingNew(G, a, b)
labels_pred = countingLastPaper(G, a, b)

t = np.linspace(0,20, 50)
plt.plot(t, f2(t,b))


"""



def simpleTriangleCounting( G, rs, rd ):
    """
    Correspond to algo in  
    The Geometric Block Model
    Sainyam Galhotra, Arya Mazumdar, Soumyabrata Pal, Barna Saha
    """
    remaining_nodes = list( G.nodes() )
    random.shuffle( remaining_nodes )
    starting_node = remaining_nodes[0]
    community_1 = [starting_node]
    community_2 = []

    n = nx.number_of_nodes(G)
    a = rs * n / np.log(n)
    b = rd * n / np.log(n)    
    g = np.max( [ y + np.sqrt(2*a - y) + np.sqrt(2*b-y) for y in np.arange( 0.0, 2*b, 0.001 ) ] )
    Es = min( a/2 - np.sqrt(a) , a+b - g  ) * np.log(n) / n
    Ed = (2*b + np.sqrt(6*b)) * np.log(n) / n

    while (len(remaining_nodes) > 0 ):
        u = random.choice( community_1 + community_2 )
        v = remaining_nodes[0]
        if (process1(G, u, v, Es, Ed ) ):
            if u in community_1:
                community_1.append(v)
            else:
                community_2.append(v)
        else:
            if u in community_1:
                community_2.append(v)
            else:
                community_1.append(v)
        remaining_nodes.remove(v)
        #print( len(remaining_nodes) )
        
    labels_pred = np.zeros( nx.number_of_nodes(G) )
    for i in community_1:
        labels_pred[i] = 1
    for i in community_2:
        labels_pred[i] = 2
    return labels_pred


def process1( G, u, v, Es, Ed ):
    edge = (u,v)
    count = calcmot1(G, edge)
    n = nx.number_of_nodes( G )

    if np.abs(count / n  - Es) < np.abs(count/n - Ed):
        return True
    return False




def countingUmassLastPaper(G, rs, rd):
    """
    Correspond to the algo in  
    Connectivity of Random Annulus Graphs and the Geometric Block Model.
    Strangely, does not correspond to algo sent by Umass team.
    
    Input: a GBM graph (2 unknown communities), with thresholds rs, rd
    Return the label prediction (n times 1 vector, with labels being 1 and 2)
    """
    # recall  rs = a logn /n, rd = b log n / n
    n = nx.number_of_nodes( G )
    a = n / np.log( n ) * rs
    b = n / np.log( n ) * rd
    
    t1 = optimize.bisect(f1, 0, 10*b, maxiter=5000, args = np.array([b]) )
    t2 = optimize.bisect(f2, 0, 2*b, args = np.array([b]) , maxiter = 5000 )
    #t2 = optimize.newton(g1, b, maxiter=5000,  args = np.array([b]) ) #other method
    n = G.number_of_nodes()
    Es = (2*b + t1) * np.log(n) / n
    Ed = (2*b - t2) * np.log(n) / n
    Gc = G.copy()
    for edge in G.edges:
        if not process2(G, edge, Es, Ed):
            Gc.remove_edge(edge[0], edge[1])
    
    labels_pred = np.zeros(n, dtype = int)
    k = 1
    for connected_component in nx.connected_components(Gc):
        for node in connected_component:
            labels_pred[node] = k
        k = k+1
    
    return labels_pred

def process2(G, edge, Es, Ed):
    count = calcmot1(G, edge)
    if (count/G.number_of_nodes() >= Es or count/G.number_of_nodes() <= Ed):
        return True
    else:
        return False
    
def f1(t,b):
    return (2*b+t) * np.log( (2*b+t)/(2*b) ) - t - 1

def f2(t,b):
    return (2*b-t) * np.log((2*b+t)/(2*b)) + t - 1

def g1(t,b):
    return (2*b+t)*np.log( (2*b+t) / (2*b) ) - 1

def g2(t,b):
    return 1 - (2*b-t)* np.log((2*b+t) / (2*b))

def calcmot1(Gc, edge):
    return len(set(Gc.neighbors(edge[0])).intersection(set(Gc.neighbors(edge[1]))))
