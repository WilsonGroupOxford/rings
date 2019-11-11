#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:45:06 2019

@author: matthew-bailey
"""

from typing import Dict, Sequence, NewType, Tuple, Any

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from collections import Counter

from shape import Shape, node_list_to_edges

Node = NewType('Node', Any)
Graph = NewType('Graph', Any)
Coord = NewType('Coord', np.array)


class RingFinder:
    """
    A group of subroutines to find rings in a combination
    of a networkx graph and a set of coordinates. The rings
    it identifies correspond to the faces on the polyhedron
    that this graph represents, according to Euler's formula.
    Proceeds by using a Delaunay triangulation which has
    rings well-defined by simplicies and then removes
    edges one-by-one.
    """
    def __init__(self,
                 graph: Graph,
                 coords_dict: Dict[Node, Coord],
                 cutoffs=None,
                 find_perimeter=True):
        self.graph: Graph = graph
        self.coords_dict: Dict[Node, Coord] = coords_dict

        # Tidying up stage -- remove the long edges,
        # and remove the single coordinate sites.
        self.cutoffs = cutoffs
        if cutoffs is not None:
            self.remove_long_edges(cutoffs)
        self.remove_single_coordinate_sites()
        self.removable_edges = None
        # Now triangulate the graph and do the real heavy lifting.
        self.tri_graph, self.simplices = self.triangulate_graph()
        self.current_shapes = {Shape(node_list_to_edges(simplex),
                                     self.coords_dict)
                               for simplex in self.simplices}
        self.identify_rings()
        if find_perimeter:
            self.find_perimeter_ring()

        
    def find_perimeter_ring(self):
        """
        Locates the perimeter ring of this arrangement,
        also known as the 'infinite face'.
        Must be called after we've found the other shapes,
        as we use that information to identify the perimeter ring.
        :return perimeter_rings: a set of the perimeter rings
        """
        # Count all the edges that are only used in one shape.
        # That means they're at the edge, so we can mark them 
        # as the perimeter ring.
        edge_use_count = Counter([edge for shape in self.current_shapes
                                  for edge in shape.edges])
        single_use_edges = {key for key, count in edge_use_count.items()
                       if count == 1}
        single_use_edges = frozenset(single_use_edges)
        # Turn this list of edges into a graph and
        # count how many rings are in it.
        perimeter_ring_graph = nx.Graph()
        perimeter_ring_graph.add_edges_from(single_use_edges)
        perimeter_coords = {node: self.coords_dict[node]
                            for node in perimeter_ring_graph.nodes()}
        sub_ring_finder = RingFinder(perimeter_ring_graph,
                                     coords_dict=perimeter_coords,
                                     cutoffs=None,
                                     find_perimeter=False)
        nx.draw(sub_ring_finder.tri_graph, pos=self.coords_dict)
        return {str(ring) for ring in sub_ring_finder.current_shapes}
        

    def remove_long_edges(self,
                          cutoffs: Sequence[float]):
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
        to_remove = set()
        for edge in self.graph.edges():
            pos_a = self.coords_dict[edge[0]]
            pos_b = self.coords_dict[edge[1]]
            distance = np.abs(pos_b - pos_a)
            if distance[0] > cutoffs[0]:
                to_remove.add(edge)
            elif distance[1] > cutoffs[1]:
                to_remove.add(edge)
        self.graph.remove_edges_from(to_remove)

    def triangulate_graph(self):
        """
        Constructs a Delauney triangulation
        of a set of coordinates, and returns
        it as a networkx graph.
        :param coordinates_dict: a dictionary, with key
        being a node and the value being an [x, y]
        numpy array.
        :return tri_graph: a Delaunay triangulation
        of the original graph.
        :return mapped_simplices: a list of all the
        edges making up triangular simplicies
        """

        # Turn the coordinate dictionary into
        # an array. The index of a given key
        # corresponds to its position in the
        # sorted list of keys, which is stored
        # in the index_to_key dict.
        coords_array = np.empty([len(self.coords_dict), 2])
        index_to_key = {}
        for i, key in enumerate(sorted(self.coords_dict.keys())):
            if self.coords_dict[key].shape[0] != 2:
                raise RuntimeError("Coordinates in the dictionary must be 2D.")
            index_to_key[i] = key
            coords_array[i, :] = self.coords_dict[key]

        tri_graph = nx.Graph()
        delaunay_res = Delaunay(coords_array)
        mapped_simplices = []
        for simplex in delaunay_res.simplices:
            # Convert these indicies to the same ones
            # the master graph uses, to avoid horrors.
            mapped_simplex = [index_to_key[node] for node in simplex]
            mapped_simplices.append(mapped_simplex)
            # Iterate over all the simplex edges and add them to
            # a graph.
            edges = node_list_to_edges(mapped_simplex)
            tri_graph.add_edges_from(edges)
        return tri_graph, mapped_simplices

    def remove_single_coordinate_sites(self) -> Graph:
        """
        Recursively finds all the single coordinate sites,
        and all the sites that would be single coordinate
        if that one were removed, and so on.
        Mutates the input data by deleting entries.
        :param main_graph: the networkx graph to detect single-coordinate
        nodes in
        :param coords_dict: the coordinates of the nodes, which we
        remove to make sure they don't get misused in the Delauney
        triangulation.
        :return graph: a graph minus the single coordinate notes. Note
        that this mutates the original graph, so the return value can
        be ignored.
        """
        while True:
            # Find the 0 or 1 coordinate nodes and make a list of them,
            # then remove both their entry in the graph and their
            # coordinate.
            nodes_to_remove = [item[0] for item in self.graph.degree()
                               if item[1] < 2]
            if not nodes_to_remove:
                break
            self.graph.remove_nodes_from(nodes_to_remove)
            for node in nodes_to_remove:
                del self.coords_dict[node]

    def identify_rings(self,
                       max_to_remove: int = None):
        """
        Removes the edges from a triangulated graph that do not exist
        in the original graph, identifying rings in the process.
        Start off with a set of simplices as the building blocks
        of rings.
        :param main_graph: the networkx graph to detect cycles in
        :param tri_graph: the Delauney triangulation of main_graph,
        as the same graph type.
        :param simplices: a list of tuples, each of which is three
        node ids representing a triangle.
        :param max_to_remove: the maximum number of edges to remove.
        Useful for making animations, but is None by default.
        """

        # First we need to check if there are any edges
        # that exist in the main graph that do not exist
        # in the triangulated graph, usually an indication
        # of unphysicality. However, networkx doesn't have
        # consistent ordering of edges, so we need to make it
        # insensitive to (a, b) <-> (b, a) swaps.
        main_edge_set = {frozenset(edge) for edge in self.graph.edges()}
        tri_edge_set = {frozenset(edge) for edge in self.tri_graph.edges()}

        if not main_edge_set.issubset(tri_edge_set):
            missing_edges = main_edge_set.difference(tri_edge_set)
            raise RuntimeError("There are edges in the main graph that do not " +
                               "exist in the Delauney triangulation: " +
                               f"{missing_edges}.")

        self.removable_edges = tri_edge_set.difference(main_edge_set)
        if max_to_remove is None:
            max_to_remove = len(self.removable_edges)
        # Each edge that we wish to remove belongs to one or two
        # shapes. Find the shapes it is in, and merge them.
        edges_removed = 1
        edge = self.removable_edges.pop()
        while self.removable_edges:
            edges_removed += 1
            self.remove_one_edge(edge)
            edge = self.removable_edges.pop()
            if edges_removed > max_to_remove:
                return
        self.remove_one_edge(edge)        

    def remove_one_edge(self, edge):
        shapes_with_edge = []
        for shape in self.current_shapes:
            if edge in shape:
                shapes_with_edge.append(shape)
            if len(shapes_with_edge) == 2:
                break

        if len(shapes_with_edge) == 1:
            # It's only part of one shape.
            # Scrap it.
            # TODO: this might have to change for periodic.
            self.current_shapes.remove(shapes_with_edge[0])
            return

        if len(shapes_with_edge) == 0:
            # This is a stranded edge. This means
            # something has gone horribly wrong
            # and we should bail out.
            return
            raise ValueError("Found an edge associated with no shapes. " +
                             "Did you remove all single coordinate nodes?")

        new_shape = shapes_with_edge[0].merge(shapes_with_edge[1])

        for shape in shapes_with_edge:
            self.current_shapes.remove(shape)
        self.current_shapes.add(new_shape)

    @property
    def ring_sizes(self):
        return [len(shape) for shape in self.current_shapes]

    def as_polygons(self):
        return [shape.to_polygon() for shape in self.current_shapes]


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

    ring_finder = RingFinder(G, COORDS_DICT, np.array([20.0, 20.0]))

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
    # AX.add_collection(p)
    AX.set_xlim(0, 155)
    AX.set_ylim(0, 155)
    # nx.draw_networkx_edges(ring_finder.graph, ax=AX, pos=COORDS_DICT,
    #                       edge_color="black", zorder=1000, width=5)
