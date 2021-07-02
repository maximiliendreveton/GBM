#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:56:57 2020

@author: mdreveto
"""

import networkx as nx
import numpy as np
import itertools
from scipy import optimize

################ UMass guys Old Algo        


def Umass_old_algo(G, rs, rd):
    """
    First algo sent by email to Andrei and I by Umass team.
    No modification done. Should correspond to the First paper (The Geometric Block Model), 
    but hard to analyze + slow execution time
    
    Return the label prediction (n times 1 vector with labels being 1 and 2)

    """
    N = nx.number_of_nodes(G)    
    if(rs>0.5):
        return print("rs  cannot be bigger than 0.5")

    size1 = N/2
    size2 = N/2
    G=G.to_undirected()
    #Gc=max(nx.connected_component_subgraphs(G), key=len)
    Gc=max( [G.subgraph(c).copy() for c in nx.connected_components(G)] , key=len)

    mst=nx.minimum_spanning_edges(Gc,data=False)
    edgelist=list(mst)
    #color=nx.get_node_attributes(Gc,'value')
    color={}
    for i in range(N//2):
        color[i]=0;color[(N/2)+i]=1
    G_mst=nx.Graph()
    G_mst.add_nodes_from(Gc)
    G_mst.add_edges_from(edgelist)
    cluster1=[]
    cluster0=[]
    temp=list(G_mst.nodes)
    tempdel=[]
    counter=0
    error=0
    c1=0
    c0=0
    e1=0
    e0=0
    
    err=[];precision_1=[];recall_1=[]

    while(len(temp)>0):
        #print (len(temp))
        counter+=1
        process_node=temp[0]
        temp.remove(process_node)
        tempdel.append(process_node)
        if (len(cluster1)==0):
            cluster1.append(process_node)
        else:
            triangles=[]
            for nbr in Gc.neighbors(process_node):
                if(nbr in cluster1):
                    edge=(nbr,process_node)
                    #print ("A",color[nbr],color[node],nbr,calcmot1(edge),motif1(rs,rd,size1,size2))
                    if (calcmot1(Gc, edge)>=motif1(rs,rd,size1,size2)):
                        triangles.append(1) ; #print (color[node],color[nbr],1)
                    else :
                        triangles.append(-1) ; #print (color[node],color[nbr],-1)
                if(nbr in cluster0):
                    edge=(nbr,process_node)
                    #print ("B",color[nbr],color[node],nbr,calcmot1(edge),motif1(rs,rd,size1,size2))
                    if (calcmot1(Gc, edge)>=motif1(rs,rd,size1,size2)):
                        triangles.append(-1) ; #print (color[node],color[nbr],-1)
                    else :
                        triangles.append(1) ; #print (color[node],color[nbr],1) 
            if len(triangles)==0:
                continue
            if(sum(triangles)>0):
                cluster1.append(process_node)
            else:
                cluster0.append(process_node)
            #print node,color[node],sum(triangles)
            #zz=input()
        #print list(G_mst.neighbors(node)) 
        #for nbr in list(G_mst.neighbors(node)) :
        #    if nbr not in tempdel:
        #        temp.append(nbr) 
        #print (color,cluster0,process_node)                          
        if (color[process_node]==1 and (process_node in cluster0)) or (color[process_node]==0 and (process_node in cluster1)):
            error+=1   
            #print (error,counter)
        if (color[process_node]==1 and (process_node in cluster1)):
            c1+=1
        elif (color[process_node]==0 and (process_node in cluster0)):
            c0+=1
        elif (color[process_node]==0 and (process_node in cluster1)):
            e1+=1
        else:
            e0+=1
    #print (error,counter)
    
    tp = (c1*(c1-1) + c0*(c0-1) + e1*(e1-1) + e0*(e0-1))/2.0
    fp = c1*e1 + c0*e0
    #print (c1, c0, e1, e0)
    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(size1*(size1-1))
    err.append(min((error*1.0)/N,1-(error*1.0)/N))
    precision_1.append(precision)
    recall_1.append(recall)
    
    labels_pred = np.ones(N)
    for i in cluster1:
        labels_pred[i] = 2
    
    return labels_pred
    #return (precision, recall, error, labels_pred)






################ UMASS guys New algo

dict={}
def findsubsets(S,m):
    return set(itertools.combinations(S, m))

        
#   print "different"
def motif1(rs,rd,size1,size2):
    #Max note: correspond to the expectation of common neighbors between two nodes;
    #SameExpectation : if the two nodes are in the same cluster;
    #DiffExpectation : the two nodes are in different cluster.
    SameExpectation = 0.0;DiffExpectation = 0.0
    if(rs >= 2*rd):
        SameExpectation = 3*rs*0.5*(size1) + size2* 2*rd*rd*1.0/rs
        DiffExpectation = (size1 + size2)*2*rd
    else:
        SameExpectation = 3*rs*0.5*(size1) + size2* (2*rd - rs*0.5)
        DiffExpectation = (size1+size2)*(2*rs - rs*rs*0.5*1/rd)
    return (SameExpectation + DiffExpectation)*0.5

def motif2(rs,rd,size1,size2):
    SameExpectation = 0.0;DiffExpectation = 0.0
    if(rs >= 2*rd):
        SameExpectation = rs*(size1) + size2*4*rd*(rs-rd)*1.0/rs
        DiffExpectation = (size1 + size2)*(2*rs- 2*rd)
    else:
        SameExpectation = (size1 + size2)* rs
        DiffExpectation = (size1+size2)*(2*(rs-rd)*(rs-rd)/rd +(2*rd-rs)*(3*rs-2*rd)/(2*rd) + (2*rd - rs)*(2*rd-rs)/(2*rd))
    return (SameExpectation + DiffExpectation)*0.5

def motif3(rs,rd,size1,size2):
    SameExpectation = 0.0;DiffExpectation = 0.0
    if(rs >= 2*rd):
        SameExpectation = (1-5*rs/2)*(size1) + size2*((1-4*rd)*(rs-2*rd)/rs  + (2*rd - 6*rd*rd)/rs)
        DiffExpectation = (size1 + size2)*(1-2*rs)
    else:
        SameExpectation = size1* (1-5*rs/2) + size2*(1-2*rd-rs*0.5)
        DiffExpectation = (size1+size2)*((rs-rd)*(1-2*rs)/rd + (2*rd - rs)*(1-3*rs/2-rd)/rd)
    return (SameExpectation + DiffExpectation)*0.5

def calcmot1(Gc, edge):
    return len(set(Gc.neighbors(edge[0])).intersection(set(Gc.neighbors(edge[1]))))

def calcmot2(Gc, edge):
    summ=len(set(Gc.nodes())-set(Gc.neighbors(edge[0])).intersection(set(Gc.neighbors(edge[1]))))
    summ+=len(set(Gc.nodes())-set(Gc.neighbors(edge[1])).intersection(set(Gc.neighbors(edge[0]))))
    return summ

def calcmot3(Gc, edge):
    return len(set(Gc.nodes())-set(Gc.neighbors(edge[0])).intersection(set(Gc.nodes())-set(Gc.neighbors(edge[1]))))


def Umass_second_algo(G, rs, rd):
    """
    Second algo send by Umass guys to Andrei and me by email.
    It doesn't correspond to any of the 2 algo in the papers, but instead it is a mix of the first paper algo and the second paper one
    """
    N = nx.number_of_nodes(G)
    err=[];
    precision_0=[];recall_0=[];precision_1=[];recall_1=[];edge_error=[]
    if(rs>0.5):
        return print("rs  cannot be bigger than 0.5")
    
    size1 = N//2
    size2 = N//2

    #Gc=max(nx.connected_component_subgraphs(G), key=len)
    Gc=max( [G.subgraph(c).copy() for c in nx.connected_components(G)] , key=len)
    #mst=nx.minimum_spanning_edges(Gc,data=False)
    #edgelist=list(mst)
    #print("starting")
    #color=nx.get_node_attributes(Gc,'value')

    G_mst=nx.Graph()
    G_mst.add_nodes_from(Gc)
    
    #G_mst.add_edges_from(edgelist)
    cluster1=[]
    cluster0=[]
    temp=[G_mst.nodes()[0]]
    tempdel=[]
    undecided=[]
    counter=0
    error=0
    c1=0
    c0=0
    e1=0
    e0=0
    
    for ed in Gc.edges():
        if (calcmot1(Gc, ed)>=motif1(rs,rd,size1,size2)):
            G_mst.add_edge(ed[0],ed[1])
    
    #print (nx.number_connected_components(G_mst))
    comp = list(nx.connected_components(G_mst))
    labels_pred = np.zeros(N, dtype = int)
    for i in range(nx.number_connected_components(G_mst)):
        for elt in comp[i]:
            labels_pred[elt] = i+1 #The community labels are 1, 2, 3 etc.
    return labels_pred
