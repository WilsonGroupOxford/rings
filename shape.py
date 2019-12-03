#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:42:41 2019

@author: matthew-bailey
"""

from typing import Dict, Sequence, NewType, Tuple, Any
from matplotlib.patches import Polygon
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
Node = NewType('Node', int)
Coord = NewType('Coord', np.array)


def calculate_polygon_area(node_list: Sequence[Node],
                           coords_dict: Dict[Node, Coord]) -> float:
    """
    Calculates the signed area of this polygon,
    using the Shoelace algorithm.
    :param node_list: an ordered list of connected nodes,
    can be clockwise or anticlockwise.
    :param coords_dict: a dictionary, keyed by nodes,
    that has [x, y] coordinates as values.
    :return signed_area: The area of the polygon,
    which is negative if the points are ordered clockwise
    and positive if the points are ordered anticlockwise.
    """
    signed_area = 0.0
    for i, node in enumerate(node_list):
        this_coord = coords_dict[node]
        next_node = node_list[(i + 1) % len(node_list)]
        next_coord = coords_dict[next_node]
        signed_area += (this_coord[0] * next_coord[1]
                        - this_coord[1] * next_coord[0])
    return 0.5 * signed_area


def node_list_to_edges(node_list: Sequence[Node],
                       is_ring: bool = True):
    """
    Takes a list of connected nodes, such that node[i] is connected
    to node[i - 1] and node[i + 1] and turn it into a set of edges.
    This is the opposite function to Shape.to_node_list
    :param node_list: a list of nodes which
    must be a hashable type.
    :param is_ring: is this a linked ring, i.e.
    is node[-1] connected to node[0]
    :return edges: a set of frozensets, each frozenset
    containing two edges.
    """
    list_size = len(node_list)
    edges = set()

    # If this is a ring, iterate one over the size
    # of the list. If not, make sure to stop
    # before the end.
    if is_ring:
        offset = 0
    else:
        offset = -1
    for i in range(list_size + offset):
        next_index = (i + 1) % list_size
        edges.add(frozenset([node_list[i],
                             node_list[next_index]]))
    return frozenset(edges)



class Shape:
    def __init__(self,
                 edges: Sequence[Tuple[Node, Node]],
                 coords_dict: Dict[Node, Coord] = None,
                 is_self_interacting: bool = False):
        """
        Initialise the shape by describing its edges, and optionally,
        their positions. If positions are not specified, this
        shape is abstract.
        :param edges: a sequence of edge tuples, either sorted
        consistently or order-independent.
        :param coords_dict: optionally a dictionary keyed by
        nodes and returning coordinates, which helps calculate
        area and winding order.
        """
        self.edges = frozenset(edges)
        self.coords_dict = coords_dict
        self._area = None
        self._is_self_interacting = is_self_interacting

    def merge(self, other, edge=None) -> None:
        """
        Merges two shapes together, removing their common edges.
        We cannot merge two shapes that believe the nodes are
        in different locations, so raises a ValueError
        if self.coords_dict[node] != other.coords_dict[node]
        for a common node.
        :param other: the shape to merge in to this one.
        :param edge: the edge to merge along. If none, we will remove all common edges, which might have some downsides!
        :return new_shape: the shape described by these
        two shapes merging together, removing the common edge
        e.g. two squares merge to form a hexagon.
        """
        if edge is None:
            unique_edges = self.edges.symmetric_difference(other.edges)
        else:
            unique_edges = set(self.edges.union(other.edges))
            unique_edges.remove(edge)
            unique_edges = frozenset(unique_edges)
 
        common_edges = self.edges.intersection(other.edges)
        if len(common_edges) >= 2:
            is_self_interacting = True
        elif other._is_self_interacting or self._is_self_interacting:
            is_self_interacting = True
        else:
            is_self_interacting = False

        common_nodes = self.nodes.intersection(other.nodes)
        for node in common_nodes:
            if not np.all(self.coords_dict[node] == other.coords_dict[node]):
                raise ValueError("These two shapes believe that node " + 
                                 f"{node} is in two different places.")
        
        new_shape = Shape(unique_edges, coords_dict=self.coords_dict, is_self_interacting=is_self_interacting)
        return new_shape

    @property
    def nodes(self):
        """
        Finds a set of the unique nodes in this shape.
        :return nodes: a set of the nodes in this shape.
        """
        nodes = {node for edge in self.edges
                 for node in edge}
        return nodes

    @property
    def area(self):
        """
        Returns the unsigned area of the ring, using
        a cached value if possible.
        """
        if self._area is None:
            self.area = np.abs(calculate_polygon_area(self.to_node_list, self.coords_dict))
        return self._area

    def to_node_list(self):
        """
        Turns the set of edges into an ordered list. e.g.
        the triangle {{0, 1}, {1, 2}, {2, 0}} becomes
        [0, 1, 2]. It puts the minimum indexed node first
        for consistent ordering. If we have coordinate information,
        this will apply a consistent anticlockwise winding to
        the nodes. If we do not have coordination information,
        this applies an ordering starting at the minimum id
        node and stepping to the next smallest numbered node.
        :return node_list: a connection ordered list of nodes.
        """
     
        if self._is_self_interacting:
            # More generally, we can find an Eulerian path.
            # This is hyper slow, so avoid it if at all possible.
            # It is only necessary in the case of a self-interacting
            # ring which shares an edge with itself.
            # TODO: I believe all non-Eulerian cycles get split up
            # in the ring finding process. Can that be proven?
            ring_graph = nx.Graph()
            ring_graph.add_edges_from(self.edges)
            odd_nodes = [node for node in ring_graph.nodes()
                         if len(list(ring_graph.neighbors(node))) % 2 == 1]
            if odd_nodes:
                start_node = min(odd_nodes)
            else:
                start_node = min(self.nodes)
            euler_path = nx.algorithms.euler.eulerian_path(G=ring_graph, source=start_node)
            node_list = [edge[0] for edge in euler_path]
            node_list = node_list + [node_list[0]]
        else:
            node_list = [min(self.nodes)]
            seen_nodes = set(node_list)
            while len(node_list) < len(self.edges):
                last_node = node_list[-1]
                # Find the two nodes this is connected to.
                connected_nodes = set()
                for edge in self.edges:
                    if last_node in edge:
                        connected_nodes = connected_nodes.union(edge)
                connected_nodes = connected_nodes.difference(seen_nodes)
                # Pick the smallest node to move to next, arbitrarily.
                # We'll sort out winding later.
                if len(connected_nodes) == 0:
                    # This is a line, not a ring! Just dump all the nodes
                    # and pray.
                    return list(self.nodes)
                next_node = min(connected_nodes)
                node_list.append(next_node)
                seen_nodes = set(node_list)

        if self.coords_dict is not None:
            signed_area = calculate_polygon_area(node_list, self.coords_dict)
            self._area = np.abs(signed_area)
            if signed_area < 0:
                # If the signed area is negative, then the ordering
                # is wrong. That's easily fixed by reversing the list,
                # and then putting the smallest element at the front.
                node_list = list(reversed(node_list))
                node_list = node_list[-1:] + node_list[:-1]
        return node_list

    def to_polygon(self):
        """
        Turns this shape into a matplotlib polygon object.
        :return polygon: a matplotlib polygon object for plotting.
        """
        if self.coords_dict is None:
            raise ValueError("self.coords_array is None, so we cannot " +
                             "construct a matplotlib polygon.")
        node_list = self.to_node_list()
        coord_array = np.empty([len(node_list), 2], dtype=float)
        for i, node in enumerate(node_list):
            coord_array[i, :] = self.coords_dict[node]
        return Polygon(coord_array, closed=True)

    def __contains__(self, obj) -> bool:
        """
        Override the in / not in magic method, because this shape is
        solely defined by its edges. If an edge is in this shape,
        return True.
        """
        return obj in self.edges

    def __hash__(self) -> int:
        """
        Override the hash magic method, because this shape is
        solely defined by its edges. This means that shapes
        in any rotation or order of edges hash the same.
        """
        return hash(self.edges)

    def __eq__(self, other) -> bool:
        """
        Override the equals magic method, because this shape is
        solely defined by its edges. This means that shapes
        in any rotation or order of edges hash the same.
        """
        return self.edges == other.edges

    def __str__(self) -> str:
        """
        Override the string magic method to make a pretty
        output.
        """
        return str(self.to_node_list())

    def __len__(self) -> int:
        """
        Override the length magic method to return the
        size of the shape
        """
        return len(self.edges)

class Line(Shape):
    def __init__(self,
                 edges: Sequence[Tuple[Node, Node]],
                 coords_dict: Dict[Node, Coord] = None):
        super().__init__(self, edges, coords_dict)

    @property
    def area(self):
        raise AttributeError("Lines do not meaningfully have an area")
