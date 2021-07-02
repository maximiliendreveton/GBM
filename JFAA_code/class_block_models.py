#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:57:33 2020

@author: mdreveto
"""

import networkx as nx
import random as rdm

class GraphParameterException (TypeError):
    def __init__(self, message):
        super().__init__(message)    
        
def isInstanceListOfIntegers( liste ):
    if not isinstance(liste, list):
        return False
    else:
        for element in liste:
            if not isinstance(element, int):
                return False
    return True



class BlockModels( nx.Graph ):
    
    def __init__(self, sizes = [250, 250], random_ordering_communities = False ):
        if not isInstanceListOfIntegers(sizes):
            raise GraphParameterException('The sizes should be given as a list of integers')

        nx.Graph.__init__(self)
        super().add_nodes_from( [i for i in range( sum(sizes) ) ] )
        self.__number_of_communities = len(sizes)
        self.__community_sizes = sizes
        labels_true = []
        for k in range(len(sizes)):
            labels_true += [ k+1] * sizes[ k ] 
            
        if(random_ordering_communities):
            rdm.shuffle( labels_true )
        for i in self.nodes:
            self.nodes[i]['community'] = labels_true [ i ]

        self.__community_ground_truth = labels_true
        
        
        
    def getLabels(self):
        dict_communinity_labels = {}
        for node in self.nodes:
            dict_communinity_labels[node] = self.nodes[node]['community']
        return dict_communinity_labels
    
    @property  
    def community_sizes(self):
        return self.__community_sizes
    
    @property  
    def number_of_communities(self):
        return self.__number_of_communities
    
    @property  
    def community_ground_truth(self):
        return self.__community_ground_truth


    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return "(N, K, sizes) = (%d, %d, %s) " % (super().number_of_nodes(), self.number_of_communities, self.community_sizes)
    

