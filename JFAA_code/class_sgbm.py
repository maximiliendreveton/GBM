#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:39:43 2020

@author: mdreveto
"""




import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import class_block_models as bm


def torusDistance(X, Y):
    return min( np.abs(X-Y), 1 - np.abs(X-Y) )



class StochasticGeometricBlockModel ( bm.BlockModels ) :
    """
    This class correspond to the StochasticGeometricBlockModel,
    with only 2 communities (sizes is given as input)
    and intra (resp. inter) connectivity function Fin (resp. Fout)
    
    """
    def __init__(self, sizes = [250, 250], fin = 0.0, fout = 0.0, random_ordering_communities = False, tqdm_ = False, distanceCutoff = 1):
        bm.BlockModels.__init__(self, sizes = sizes, random_ordering_communities = random_ordering_communities)
        self.intra_cluster_function = fin
        self.inter_cluster_function = fout
        for i in super().nodes:
            super().nodes[i]["pos"] = np.random.rand()
        
        if( distanceCutoff >= 1.0 ):
            super().add_edges_from( self.makeEdges( super().number_of_nodes(), fin, fout, self.community_ground_truth, tqdm_ ) )
        else:
            super().add_edges_from( self.makeEdges_fast( super().number_of_nodes(), fin, fout, self.community_ground_truth, tqdm_, distanceCutoff ) )

            
    def getPositions(self):
        dict_positions = {}
        for node in super().nodes:
            dict_positions[node] = super().nodes[node]['pos']
        return dict_positions
    
    
    def makeEdges(self, N, fin, fout, labels_nodes, tqdm_):
        edges = []
        if (tqdm_):
            loop = tqdm(range(N))
        else:
            loop = range( N )
        for i in loop:
            for j in range(i):
                if labels_nodes[i] == labels_nodes[j]:
                    edgeProbaFunction = fin
                else:
                    edgeProbaFunction = fout
                if ( np.random.rand( ) < edgeProbaFunction( torusDistance( super().nodes[i]["pos"] , super().nodes[j]["pos"] ) ) ):
                    edges.append( ( i,j ) )        
        return edges
    
    
    def makeEdges_fast( self, N, fin, fout, labels_nodes, tqdm_ , distanceCutoff ):
        edges = []
        positions = self.getPositions()
        nodesIndexSortedByGeography = np.argsort( [ position for position in positions.values() ] )
        if (tqdm_):
            loop = tqdm( range( N ) )
        else:
            loop = range( N )
        
        for i in loop:
            j = (i+1) % N
            while ( torusDistance( positions[ nodesIndexSortedByGeography[i] ], positions[ nodesIndexSortedByGeography[j] ] ) < distanceCutoff ) :
                if labels_nodes[i] == labels_nodes[j]:
                    edgeProbaFunction = fin
                else:
                    edgeProbaFunction = fout
                
                if ( np.random() < edgeProbaFunction( torusDistance( positions[ nodesIndexSortedByGeography[i] ], positions[ nodesIndexSortedByGeography[j] ] ) ) ):
                    edges.append( ( nodesIndexSortedByGeography[i], nodesIndexSortedByGeography[j] ) )
                j = (j+1) % N
        
        return edges

        
        
        
    
    @property  
    def intra_cluster_function(self):
        return self.__intra_cluster_function
    
    @intra_cluster_function.setter  
    def intra_cluster_function(self, newThresh):
        self.__intra_cluster_function = newThresh


    @property
    def inter_cluster_function(self):
        return self.__inter_cluster_function
    
    @inter_cluster_function.setter 
    def inter_cluster_function(self, newThresh):
        self.__inter_cluster_function = newThresh
    
    def subgraph(self, nodes):
        G_subgraph = super().subgraph(nodes)
        G_subgraph.inter_cluster_function = self.inter_cluster_function
        G_subgraph.intra_cluster_function = self.intra_cluster_function
        return G_subgraph    
    
    
    def __repr__(self):
        return self.__str__()
    
    
    def __str__(self):
        return "(N, K, sizes, rs, rd) = (%d, %d, %s, %8.3f, %8.3f) " % (super().number_of_nodes(), super().number_of_communities, super().community_sizes, self.__intra_cluster_function, self.__inter_cluster_function)

    
    def draw(self, node_color, pos=None, node_size=200, alpha_node = 1, alpha_edge = 0.01, width = 0.01, cmap=plt.cm.RdYlBu, style = 'solid'):
        if (pos == None):
            pos = {}
            for node in super().nodes():
                pos[node] = self.number_of_nodes() * 1 / (2 * np.pi) * np.array( [ np.cos(self.nodes[node]['pos'] * 2 * np.pi) ,  np.sin(self.nodes[node]['pos'] * 2 * np.pi) ])
        nx.draw_networkx_nodes(self, pos=pos, node_size = node_size, alpha = alpha_node, cmap= cmap, vmin = min(node_color), vmax = max(node_color), node_color= node_color, style = style )
        nx.draw_networkx_edges(self, pos=pos, alpha = alpha_edge, width = width, style = style )

