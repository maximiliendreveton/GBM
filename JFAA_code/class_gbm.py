#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:47:31 2020

@author: mdreveto
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import class_block_models as bm


def torusDistance(X, Y):
    return min( np.abs(X-Y), 1 - np.abs(X-Y) )



class GBM2d ( bm.BlockModels ) :
    """
    This class correspond to the GBM model in 2 dimensions,
    with only 2 communities (sizes is given as input)
    and intra (resp. inter) radius threshold rs (resp. rd)
    
    """
    def __init__(self, sizes = [250, 250], rs = 0.0, rd = 0.0, random_ordering_communities = False, fast_generation = True, tqdm_ = False):
        bm.BlockModels.__init__(self, sizes = sizes, random_ordering_communities = random_ordering_communities)
        self.intra_cluster_threshold = rs
        self.inter_cluster_threshold = rd
        for i in super().nodes:
            super().nodes[i]["pos"] = np.random.rand()
        
        if (fast_generation):
            super().add_edges_from( self.makeEdges_fast( super().number_of_nodes(), rs, rd, self.community_ground_truth, tqdm_ ) )
        else:
            super().add_edges_from( self.makeEdges( super().number_of_nodes(), rs, rd, self.community_ground_truth, tqdm_ ) )
    
    
    def getPositions(self):
        dict_positions = {}
        for node in super().nodes:
            dict_positions[node] = super().nodes[node]['pos']
        return dict_positions
    
    
    def makeEdges(self, n, rs, rd, labels_nodes, tqdm_):
        edges = []
        cutoffMatrix = np.array( [ [rs,rd], [rd,rs] ])
        if (tqdm_):
            loop = tqdm(range(n))
        else:
            loop = range( n )
        for i in loop:
            for j in range(i):
                if( np.min( [np.abs(super().nodes[i]["pos"] - super().nodes[j]["pos"]), 1.0 - np.abs( super().nodes[i]["pos"] - super().nodes[j]["pos"] ) ] ) <  cutoffMatrix[labels_nodes[i]-1, labels_nodes[j]-1  ]  ):
                #if (torusDistance( self.nodes[i]["pos"], self.nodes[j]["pos"] ) <  cutoffMatrix[labels_nodes[i], labels_nodes[j]]  ):
                    edges.append( (i,j) )        
        return edges
    
    
    def makeEdges_fast( self, n, rs, rd, labels_nodes, tqdm_ ):
        edges = []
        cutoffMatrix = np.array([[rs,rd], [rd,rs]])
        positions = self.getPositions()
        nodesIndexSortedByGeography = np.argsort( [ position for position in positions.values() ] )
        if (tqdm_):
            loop = tqdm( range(n) )
        else:
            loop = range(n)
        
        for i in loop:
            j = (i+1) % n
            while ( torusDistance( positions[ nodesIndexSortedByGeography[i] ], positions[ nodesIndexSortedByGeography[j] ] ) < max(rs, rd) ) :
                if( torusDistance( positions[ nodesIndexSortedByGeography[i] ], positions[ nodesIndexSortedByGeography[j] ] ) < cutoffMatrix[labels_nodes[ nodesIndexSortedByGeography[i] ] - 1, labels_nodes[ nodesIndexSortedByGeography[j] ] - 1  ] ) :
                    edges.append( ( nodesIndexSortedByGeography[i], nodesIndexSortedByGeography[j] ) )
                j = (j+1) % n
        return edges
        
    
    @property  
    def intra_cluster_threshold(self):
        return self.__intra_cluster_threshold
    
    @intra_cluster_threshold.setter  
    def intra_cluster_threshold(self, newThresh):
        if not isinstance(newThresh, float):
            raise bm.GraphParameterException("Intra cluster threshold must be a float")
        elif newThresh < 0:
            raise bm.GraphParameterException("Intra cluster threshold cannot be negative")
        else:
            self.__intra_cluster_threshold = newThresh


    @property
    def inter_cluster_threshold(self):
        return self.__inter_cluster_threshold
    
    @inter_cluster_threshold.setter 
    def inter_cluster_threshold(self, newThresh):
        if not isinstance(newThresh, float):
            raise bm.GraphParameterException("Inter cluster threshold must be a float")
        elif newThresh < 0:
            raise bm.GraphParameterException("Inter cluster threshold cannot be negative")
        else:
            self.__inter_cluster_threshold = newThresh
    
    def subgraph(self, nodes):
        G_subgraph = super().subgraph(nodes)
        G_subgraph.inter_cluster_threshold = self.inter_cluster_threshold
        G_subgraph.intra_cluster_threshold = self.intra_cluster_threshold
        return G_subgraph    
    
    
    def __repr__(self):
        return self.__str__()
    
    
    def __str__(self):
        return "(N, K, sizes, rs, rd) = (%d, %d, %s, %8.3f, %8.3f) " % (super().number_of_nodes(), super().number_of_communities, super().community_sizes, self.__intra_cluster_threshold, self.__inter_cluster_threshold)

    
    def draw(self, node_color, pos=None, node_size=200, alpha_node = 1, alpha_edge = 0.01, width = 0.01, cmap=plt.cm.RdYlBu, style = 'solid'):
        if (pos == None):
            pos = {}
            for node in super().nodes():
                pos[node] = self.number_of_nodes() * 1 / (2 * np.pi) * np.array( [ np.cos(self.nodes[node]['pos'] * 2 * np.pi) ,  np.sin(self.nodes[node]['pos'] * 2 * np.pi) ])
        nx.draw_networkx_nodes(self, pos=pos, node_size = node_size, alpha = alpha_node, cmap= cmap, vmin = min(node_color), vmax = max(node_color), node_color= node_color )
        nx.draw_networkx_edges(self, pos=pos, alpha = alpha_edge, width = width )

"""
def draw(G):
    pos = {}
    for node in G.nodes():
        pos[node] = np.array([ np.cos(G.nodes[node]['pos'] * 2 * np.pi) ,  np.sin(G.nodes[node]['pos'] * 2 * np.pi) ])
    return pos

nx.draw_networkx(G, pos=pos, node_size=10, alpha = 0.1, width = 0.1)
"""

"""
f = plt.figure()
G.draw(node_color=labels_pred, node_size=50, alpha_node=0.2, alpha_edge= 0, width=0)
f.savefig("graph.eps", format = 'eps')
"""



class GBM_mean_field_old ( GBM2d ):
    
    def __init__(self, n = 500, rs = 0.0, rd = 0.0, fast_generation = True, tqdm_ = False):
        nx.Graph.__init__(self)
        super().add_nodes_from( [i for i in range(n)] )
        self.intra_cluster_threshold = rs
        self.inter_cluster_threshold = rd
        for i in self.nodes:
            if i % 2 == 0:
                self.nodes[i]['community'] = 1
            else:
                self.nodes[i]['community'] = 2
        self.__community_ground_truth = self.getLabels()
        for i in self.nodes:
            self.nodes[i]["pos"] = i / n
        if (fast_generation):
            super().add_edges_from( self.makeEdges_fast(n, rs, rd, self.community_ground_truth, tqdm_ ) )
        else:
            super().add_edges_from( self.makeEdges(n, rs, rd, self.community_ground_truth, tqdm_ ) )
            
    @property  
    def community_ground_truth(self):
        return self.__community_ground_truth




"""

class GBM_mean_field_2_communities( bm.BlockModels ):
    
    def __init__ (self, sizes, rs = 0.0, rd = 0.0, fast_generation = True, tqdm_ = False):
        
        
"""