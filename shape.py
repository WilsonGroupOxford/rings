#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:42:41 2019

@author: matthew-bailey
"""

from typing import Any, Dict, NewType, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import deque
from matplotlib.patches import Polygon
import copy
import scipy.spatial

Node = NewType("Node", int)
Coord = NewType("Coord", np.array)



def calculate_polygon_area(
    node_list: Sequence[Node], coords_dict: Dict[Node, Coord]
) -> float:
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
        signed_area += this_coord[0] * next_coord[1] - this_coord[1] * next_coord[0]
    return 0.5 * signed_area


def node_list_to_edges(node_list: Sequence[Node], is_ring: bool = True):
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
        if node_list[i] != node_list[next_index]:
            edges.add(frozenset([node_list[i], node_list[next_index]]))
    return frozenset(edges)


class Shape:
    def __init__(
        self,
        edges: Sequence[Tuple[Node, Node]],
        coords_dict: Dict[Node, Coord] = None,
        is_self_interacting: bool = False,
    ):
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
        self.edges = frozenset(copy.deepcopy(edges))
        self.coords_dict = coords_dict
        self._area = None
        self._is_self_interacting = is_self_interacting
        self._node_list = None

    def bounding_box(self) -> np.array:
        """
        Calculate the minimum bounding box for this polygon.

        :return bounding_box: a rectangle of coordinates[[min_x, max_x], [min_y, max_y]] that contains the shape
        """
        mins = np.array([np.inf, np.inf])
        maxes = np.array([-np.inf, -np.inf])
        for node in self.nodes:
            node_pos = self.coords_dict[node]
            mins = np.minimum(mins, node_pos)
            maxes = np.maximum(maxes, node_pos)
        return np.vstack([mins, maxes]).T

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
                raise ValueError(
                    "These two shapes believe that node "
                    + f"{node} is in two different places."
                )

        new_shape = Shape(
            unique_edges,
            coords_dict=self.coords_dict,
            is_self_interacting=is_self_interacting,
        )
        return new_shape

    def normal_vector(self, embedding=None):
        if embedding is None:
            embedding = self.coords_dict
        mean_normal = np.zeros(3, dtype=float)
        node_list = self.to_node_list()
        for idx in range(len(node_list)):
            node = node_list[idx]
            dim = len(embedding[node])
            neighbours = node_list[idx - 1], node_list[(idx + 1) % len(node_list)]
            vec_a, vec_b = np.zeros(3, dtype=float), np.zeros(3, dtype=float)
            vec_a[:dim] = embedding[neighbours[0]] - embedding[node]
            vec_b[:dim] = embedding[neighbours[1]] - embedding[node]

            cross = np.cross(vec_a, vec_b)
            cross /= np.linalg.norm(cross)
            mean_normal += cross
        mean_normal /= np.linalg.norm(mean_normal)
        return mean_normal


    @property
    def nodes(self):
        """
        Finds a set of the unique nodes in this shape.
        :return nodes: a set of the nodes in this shape.
        """
        nodes = {node for edge in self.edges for node in edge}
        return nodes

    def convex_hull_area(self):
        """
        Calculates the area of the convex hull of this polygon.
        """
        _coords_arr = np.vstack([self.coords_dict[node] for node in self.nodes])
        return float(scipy.spatial.ConvexHull(_coords_arr).volume)

    def convex_hull_perimeter(self):
        """
        Calculates the perimeter of the convex hull of this polygon.
        """
        _coords_arr = np.vstack([self.coords_dict[node] for node in self.nodes])
        # careful of scipy horror -- area is perimeter, and this is undocumented.
        return float(scipy.spatial.ConvexHull(_coords_arr).area)

    def solidity_metric(self):
        """
        Calculate a metric between 0 and 1 representing how solid this polygon is/

        This is calculated as the ratio between the current
        area and the area of the convex hull of these points.
        In case of numerical issues, clips to being between 0 and 1.
        """
        return np.clip(self.area / self.convex_hull_area(), 0, 1)

    def convexity_metric(self):
        """
        Calculate a metric between 0 and 1 representing how convex this polygon is.
        this is calculated as the ratio between the current
        perimeter and the perimeter of the convex hull of these points.
        In case of numerical issues, clips to being between 0 and 1.
        """
        # print(f"Our perimeter is {self.perimeter}, the convex hull perimeter is {self.convex_hull_perimeter()}")
        return np.clip(self.convex_hull_perimeter() / self.perimeter, 0, 1)

    def balanced_repartition_metric(self) -> float:
        """
        Calculates how even the shape is in the x and y directions.

        See
        'Robust shape regularity criteria for superpixel evaluation'
        Giraud, Remi and Ta, Vinh Thong and Papadakis, Nicolas
        Proceedings - International Conference on Image Processing, ICIP
        September 2017
        """
        _coords_arr = np.vstack([self.coords_dict[node] for node in self.nodes])
        coords_std = np.std(_coords_arr, axis=0, ddof=1)
        return np.sqrt(np.min(coords_std) / np.max(coords_std))

    def regularity_metric(self) -> float:
        return self.balanced_repartition_metric() * self.convexity_metric() * self.solidity_metric()

    @property
    def perimeter(self):
        perimeter = 0.0
        for edge in self.edges:
            edge = tuple(edge)
            diff = self.coords_dict[edge[1]] - self.coords_dict[edge[0]]
            length = np.hypot(*diff)
            perimeter += length
        return perimeter

    @property
    def area(self):
        """
        Returns the unsigned area of the ring, using
        a cached value if possible.
        """
        if self._area is None:
            self._area = np.abs(
                calculate_polygon_area(self.to_node_list(), self.coords_dict)
            )
        return self._area

    def centroid(self):
        """
        Returns the position of the centre-of-mass of the polygon.
        """
        nodes = self.to_node_list()
        centroid = np.array([0.0, 0.0])
        for i, node in enumerate(nodes):
            next_node = nodes[(i + 1) % len(nodes)]
            cross_term = (
                self.coords_dict[node][0] * self.coords_dict[next_node][1]
                - self.coords_dict[next_node][0] * self.coords_dict[node][1]
            )
            centroid[0] += (
                self.coords_dict[node][0] + self.coords_dict[next_node][0]
            ) * cross_term
            centroid[1] += (
                self.coords_dict[node][1] + self.coords_dict[next_node][1]
            ) * cross_term
        centroid /= 6 * self.area
        return centroid

    def _eulerian_node_list(self, ring_graph):
        """
        Calculate the node list using an Eulerian path method.

        This is slower than doing it the fast way, but a bit more
        reliable in the case of singly-self interacting rings.
        Returns
        -------
        None.

        """
        # More generally, we can find an Eulerian path.
        # This is hyper slow, so avoid it if at all possible.
        # It is only necessary in the case of a self-interacting
        # ring which shares an edge with itself.
        odd_nodes = [
            node
            for node in ring_graph.nodes()
            if len(list(ring_graph.neighbors(node))) % 2 == 1
        ]
        if odd_nodes:
            start_node = min(odd_nodes)
        else:
            start_node = min(self.nodes)
        euler_path = nx.algorithms.euler.eulerian_path(G=ring_graph, source=start_node)
        node_list = [edge[0] for edge in euler_path]
        node_list = node_list + [node_list[0]]
        return node_list

    def shared_edges(self, other_shape, num_nodes):
        self_modulo_edges = set(
            frozenset([tuple(edge)[0] % num_nodes, (tuple(edge)[1] % num_nodes)])
            for edge in self.edges
        )
        other_modulo_edges = set(
            frozenset([tuple(edge)[0] % num_nodes, (tuple(edge)[1] % num_nodes)])
            for edge in other_shape.edges
        )
        return len(self_modulo_edges.intersection(other_modulo_edges))

    def _bridges_node_list(self, ring_graph):
        """
        Calculate the node list using a bridge splitting method.

        This is useful for graphs with enclosures or exclaves,
        and can deal with them generally but it is very slow.

        """
        bridges = list(nx.bridges(ring_graph))
        ring_graph.remove_edges_from(bridges)

        # Now split the ring graph up into its connected components.
        # Then, turn each of them into a shape and repeat this
        # sorry process.
        node_list = []
        components_to_visit = list(nx.connected_components(ring_graph))
        components_to_visit.sort(key=len, reverse=False)
        while components_to_visit:
            component = components_to_visit.pop()
            if len(component) == 1:
                # This is a loose node.
                # Pop it and go about our merry way.
                continue
            edges_in_component = set(
                edge
                for edge in self.edges
                if list(edge)[0] in component and list(edge)[1] in component
            )
            # Do we need to pass on is_self_interacting? Its's slower to do so
            # because we have to find the Eulerian path, but probably safer.
            this_sub_ring = Shape(
                edges_in_component, self.coords_dict, is_self_interacting=False
            )
            sub_ring_node_list = this_sub_ring.to_node_list()
            # We've only found one component, so make that the base of our node list.
            if not node_list:
                node_list = sub_ring_node_list
                continue
            else:
                # Find the bridges connecting this connected component to the rest of
                # the graph. There can be many, so follow each of them to their ends.
                added_bridge = False
                for bridge in bridges:
                    bridge_path = []
                    seen_nodes = set()
                    if bridge[0] in sub_ring_node_list:
                        bridge_path = [bridge[0], bridge[1]]
                    elif bridge[1] in sub_ring_node_list:
                        bridge_path = [bridge[1], bridge[0]]
                    else:
                        # We're not connected to this bridge.
                        # Carry on merrily.
                        continue

                    seen_nodes.update(bridge)
                    while True:
                        path_updated = False
                        for other_bridge in bridges:
                            if (
                                bridge_path[-1] == other_bridge[0]
                                and other_bridge[1] not in seen_nodes
                            ):
                                bridge_path.append(other_bridge[1])
                                seen_nodes.update(other_bridge)
                                path_updated = True
                            elif (
                                bridge_path[-1] == other_bridge[1]
                                and other_bridge[0] not in seen_nodes
                            ):
                                bridge_path.append(other_bridge[0])
                                seen_nodes.update(other_bridge)
                                path_updated = True
                        if not path_updated:
                            # We've completed this path
                            break
                    # Rotate our node list to start at bridge_path[0]
                    rotation_index = sub_ring_node_list.index(bridge_path[0])
                    sub_ring_node_list = deque(sub_ring_node_list)
                    sub_ring_node_list.rotate(-rotation_index)
                    sub_ring_node_list = list(sub_ring_node_list)
                    # Check this bridge ends in the current connected component
                    if bridge_path[-1] in node_list:
                        insertion_pos = node_list.index(bridge_path[-1])
                        bridging_node_list = (
                            bridge_path[::-1]
                            + sub_ring_node_list[1:]
                            + bridge_path[:-1]
                        )
                        node_list = (
                            node_list[:insertion_pos]
                            + bridging_node_list
                            + node_list[insertion_pos:]
                        )
                        added_bridge = True
                        break
                if not added_bridge:
                    # We didn't succesfully bridge this. Add it
                    # back to the pile and carry on.
                    components_to_visit.insert(0, component)
        return node_list

    def to_node_list(self):
        """
        Turns the set of edges into an ordered list.

        e.g. the triangle {{0, 1}, {1, 2}, {2, 0}} becomes
        [0, 1, 2]. It puts the minimum indexed node first
        for consistent ordering. If we have coordinate information,
        this will apply a consistent anticlockwise winding to
        the nodes. If we do not have coordination information,
        this applies an ordering starting at the minimum id
        node and stepping to the next smallest numbered node.

        This is used to calculate the area of shapes and to draw them.

        It is memoised, so be careful if you change shape.edges.
        :return node_list: a connection ordered list of nodes.
        """

        if self._node_list is not None:
            return self._node_list

        if self._is_self_interacting:

            ring_graph = nx.Graph()
            ring_graph.add_edges_from(self.edges)

            if nx.is_eulerian(ring_graph):
                # More generally, we can find an Eulerian path.
                # This is hyper slow, so avoid it if at all possible.
                # It is only necessary in the case of a self-interacting
                # ring which shares an edge with itself.
                node_list = self._eulerian_node_list(ring_graph)
            elif nx.has_bridges(ring_graph):
                # Except in hyper-pathological cases, where a ring is doubly
                # self interacting or has exclaves / enclaves. In this case,
                # we can identify bridges as being the edges to these exclaves.
                node_list = self._bridges_node_list(ring_graph)
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
                if len(connected_nodes) == 0:
                    # Our self-interacting detection heuristics have failed.
                    # We've tried to do it the fast way, but we can't.
                    # Restart this process and do it the slow way.
                    self._is_self_interacting = True
                    return self.to_node_list()

                # Pick the smallest node to move to next, arbitrarily.
                # We'll sort out winding later.
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
        self._node_list = node_list
        return self._node_list

    def to_polygon(self):
        """
        Turn this shape into a matplotlib polygon object.
        :return polygon: a matplotlib polygon object for plotting.
        """
        if self.coords_dict is None:
            raise ValueError(
                "self.coords_array is None, so we cannot "
                + "construct a matplotlib polygon."
            )
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
    def __init__(
        self, edges: Sequence[Tuple[Node, Node]], coords_dict: Dict[Node, Coord] = None
    ):
        super().__init__(self, edges, coords_dict)

    @property
    def area(self):
        raise AttributeError("Lines do not meaningfully have an area")

def generate_regular_polygon(num_sides:int, side_length:float=1.0) -> Shape:
    """
    Generate a regular polygon with num_sides sides, centered at 0, 0.

    :param num_sides: the number of sides the polygon has
    :param side_length: the length of each side
    :return: a shape with num_sides sides
    """
    coords = {i: np.array([side_length * np.cos(2*np.pi * i / num_sides),
                           side_length * np.sin(2*np.pi * i / num_sides)])
              for i in range(num_sides)}
    edges = [(i, (i + 1) % num_sides) for i in range(num_sides)]
    return Shape(edges, coords)

if __name__ == "__main__":
    COORDS = {0: np.array([0.0, 0.0]),
              1: np.array([1.0, 0.0]),
              2: np.array([1.0, 1.0])}
    TRIANGLE = Shape([(0, 1), (1, 2), (2, 0)], COORDS)
    print(TRIANGLE.area, TRIANGLE.convex_hull_area(), TRIANGLE.solidity_metric())
    print(TRIANGLE.perimeter, TRIANGLE.convex_hull_perimeter(), TRIANGLE.convexity_metric())
    for NUM_SIDES in range(3, 10):
        SHAPE = generate_regular_polygon(NUM_SIDES)
        print(SHAPE.edges, SHAPE.regularity_metric())

