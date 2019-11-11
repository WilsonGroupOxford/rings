#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:32:13 2019

@author: matthew-bailey
"""

from ring_finder import RingFinder
from shape import Shape, node_list_to_edges

from typing import Dict, Sequence, NewType, Tuple, Any
from matplotlib.patches import Polygon
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


Node = NewType('Node', Any)
Graph = NewType('Graph', Any)
Coord = NewType('Coord', np.array)


class PeriodicRingFinder(RingFinder):
    
    def __init__(self,
             graph: Graph,
             coords_dict: Dict[Node, Coord],
             cell=None):
        self.graph: Graph = graph
        self.coords_dict: Dict[Node, Coord] = coords_dict
        self.original_nodes = {key for key in self.coords_dict.keys()}
        # Tidying up stage -- remove the long edges,
        # and remove the single coordinate sites.
        self.cell = cell
        self.cutoffs = cell / 2.0
        self.find_periodic_edges()
        self.remove_long_edges(self.cutoffs)
        self.remove_single_coordinate_sites()
        self.removable_edges = None
        # Now triangulate the graph and do the real heavy lifting.
        self.tri_graph, self.simplices = self.triangulate_graph()
        self.current_shapes = {Shape(node_list_to_edges(simplex),
                                     self.coords_dict)
                               for simplex in self.simplices}
        self.identify_rings()
        self.current_shapes = self.find_unique_rings()
        
    def find_unique_rings(self):
        """
        Each ring in self.current_shapes has 8 periodic images.
        Identify just one of each, and keep that.
        """
        num_nodes = len(self.original_nodes)
        unique_rings = defaultdict(set)
        for ring in self.current_shapes:
            modulo_edges = set()
            for edge in ring.edges:
                modulo_edge = frozenset({item % num_nodes for item in edge})
                modulo_edges.add(modulo_edge)
            new_ring = Shape(modulo_edges)
            unique_rings[new_ring].add(ring)
        # Now, from this set of unique rings we pick just
        # one to plot -- the one that has the most nodes
        # in the original periodic image.
        canonical_rings = set()
        for unique_ring, copy_rings in unique_rings.items():
            copy_rings = list(copy_rings)
            shared_edges = []
            for copy_ring in copy_rings:
                shared_edges.append(len(unique_ring.edges.intersection(copy_ring.edges)))
            max_shared_edges = max(shared_edges)
            max_shared_idx = shared_edges.index(max_shared_edges)
            canonical_rings.add(copy_rings[max_shared_idx])
        return canonical_rings
            
    def find_periodic_edges(self):
        """
        Remove any edges that are longer than
        a set of cutoffs, useful to make a periodic cell
        aperiodic.
        :param graph: the networkx graph to detect single-coordinate
        nodes in
        :param coords_dict: a dictionary, keyed by nodes,
        with values being the [x, y] coordinates of the nodes, which
        we use to remove long bonds.
        :param cutoffs: an [max_x, max_y] sequence, removing any edges
        with a component longer than max_x or max_y. For the minimum
        image convention, we want these to be half the both length.
        :return graph: a graph minus the edges that are too long. Note
        that this mutates the original graph, so the return value can
        be ignored.
        """
        periodic_coords_dict = dict()
        num_nodes = len(self.coords_dict)
        cell_offsets = [(0, 0), (1, 1), (1, 0), (1, -1),
                        (0, 1), (0, -1),
                        (-1, 1), (-1, 0), (-1, -1)]
        # TODO: Speed this up by pre-detecting all
        # the rings in the central image and only
        # adding images of the "perimeter rings".
        # Then, analyse only the simplices that
        # we didn't account for last time.
        to_add = set()
        for node, pos in self.coords_dict.items():
            # There are 8 periodic images of this node.
            # Indexed by number of offset periodic cells
            # e.g. [+1, +1], [+1, 0], [+1, -1], [0, +1]
            # [0, -1], [-1, +1], [-1, 0], [-1, +1]
            for i, cell_offset in enumerate(cell_offsets):
                offset = np.array(cell_offset) * self.cell
                new_pos = self.coords_dict[node] + offset
                new_id = (i * num_nodes) + node
                periodic_coords_dict[new_id] = new_pos
                for neighbor in self.graph.neighbors(node):
                    to_add.add((new_id, neighbor + (i * num_nodes)))
        self.coords_dict.update(periodic_coords_dict)
           
        to_remove = set()
       
        for edge in self.graph.edges():
            pos_a = self.coords_dict[edge[0]]
            pos_b = self.coords_dict[edge[1]]
            distance = np.abs(pos_b - pos_a)
            if distance[0] > self.cutoffs[0] and distance[1] > self.cutoffs[1]:
                # Add a periodic image 
                # in +x, +y
                raise NotImplementedError("Corner links not yet implemented")
                print(pos_a, pos_b, "(+, +)")
            elif distance[1] > self.cutoffs[1]:
                # There is an edge that spans the y-coordinate.
                # Remove it, and add in the two new edges.
                to_remove.add(edge)
                
                # Find the lower and upper of the two nodes.
                lower_node = edge[np.argmin([pos_a[1], pos_b[1]])]
                upper_node = edge[np.argmax([pos_a[1], pos_b[1]])]
                
                # There are 6 connections to make here.
                for lower_offset in cell_offsets:
                    neighbor_offset = (lower_offset[0], lower_offset[1] - 1)
                    if neighbor_offset[1] < -1:
                        continue
                    
                    if lower_offset == (0, 0):
                        node_mult = 0
                    else:
                        node_mult = cell_offsets.index(lower_offset)
                    
                    if neighbor_offset == (0, 0):
                        neighbor_mult = 0
                    else:
                        neighbor_mult = cell_offsets.index(neighbor_offset)

                    to_add.add((lower_node + (node_mult * num_nodes),
                                upper_node + (neighbor_mult * num_nodes)))
                    
            elif distance[0] > self.cutoffs[0]:
                # There is an edge that spans the y-coordinate.
                # Remove it, and add in the two new edges.
                to_remove.add(edge)
                
                # Find the lower and upper of the two nodes.
                lower_node = edge[np.argmin([pos_a[0], pos_b[0]])]
                upper_node = edge[np.argmax([pos_a[0], pos_b[0]])]
                
                # There are 6 connections to make here.
                for lower_offset in cell_offsets:
                    neighbor_offset = (lower_offset[0] - 1, lower_offset[1])
                    if neighbor_offset[0] < -1:
                        continue
                    
                    if lower_offset == (0, 0):
                        node_mult = 0
                    else:
                        node_mult = cell_offsets.index(lower_offset)
                    
                    if neighbor_offset == (0, 0):
                        neighbor_mult = 0
                    else:
                        neighbor_mult = cell_offsets.index(neighbor_offset)

                    to_add.add((lower_node + (node_mult * num_nodes),
                                upper_node + (neighbor_mult * num_nodes)))
        self.graph.remove_edges_from(to_remove)
        self.graph.add_edges_from(to_add)
 
           
if __name__ == "__main__":
    G: Graph = nx.Graph()
    with open("./edges.dat", "r") as fi:
        fi.readline()  # Skip header
        for line in fi.readlines():
            x, y = [int(item) for item in line.split(",")]
            G.add_edge(x, y)

    COORDS_DICT: Dict[Node, Coord] = {}
    with open("./coords.dat", "r") as fi:
        fi.readline()  # Skip header
        for line in fi.readlines():
            line = line.split(",")
            node_id, x, y = int(line[0]), float(line[1]), float(line[2])
            COORDS_DICT[node_id] = np.array([x, y])

    ring_finder = PeriodicRingFinder(G, COORDS_DICT, np.array([150,
                                                               150]))

    FIG, AX = plt.subplots()
    FIG.patch.set_visible(False)
    AX.axis('off')
    POLYS = ring_finder.as_polygons()
    SIZES = ring_finder.ring_sizes
    SIZE_RANGE = max(SIZES) + 1 - min(SIZES)
    THIS_CMAP = plt.cm.get_cmap("viridis")(np.linspace(0, 1, SIZE_RANGE))
    COLOURS = [THIS_CMAP[SIZE - 4] for SIZE in SIZES]

    p = PatchCollection(POLYS, alpha=1, linewidth=5,
                        linestyle="dotted")
    p.set_color(COLOURS)
    p.set_edgecolor("black")
    AX.add_collection(p)
    AX.set_xlim(-150, 310)
    AX.set_ylim(-150, 310)
    nx.draw_networkx_edges(ring_finder.graph, ax=AX, pos=COORDS_DICT,
                           edge_color="black", zorder=1000, width=5)