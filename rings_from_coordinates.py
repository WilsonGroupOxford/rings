#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:45:06 2019

@author: matthew-bailey
"""

from collections import Counter, defaultdict
from typing import Dict, FrozenSet, NewType, Sequence, Set, Tuple
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
import matplotlib.colors as colors


Node = NewType("Node", int)
Graph = NewType("Graph", nx.Graph)
Coord = NewType("Coord", np.array)
Edge = NewType("Edge", FrozenSet[Tuple[Node, Node]])
ID = 0

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


class RingFinderError(Exception):
    """Exception to represent a failure to find any rings."""

    def __init__(self, message: str):
        """Initialise a default Exception object"""
        super().__init__(f"RingFinderError: {message}")


class RingFinder:
    """
    Find the rings in a planar graph.

    A group of subroutines to find rings in a combination
    of a networkx graph and a set of coordinates. The rings
    it identifies correspond to the faces on the polyhedron
    that this graph represents, according to Euler's formula.
    Proceeds by using a Delaunay triangulation which has
    rings well-defined by simplicies and then removes
    edges one-by-one.
    """

    def __init__(
        self,
        graph: Graph,
        coords_dict: Dict[Node, Coord],
        cutoffs: np.array = None,
        find_perimeter: bool = True,
        missing_policy="add",
    ):
        """
        Initialise and locate the rings in a provided graph.

        :param graph: a networkx graph object
        :param coords_dict: a dictionary of node coordinates, with ids
        corresponding to networkx node ids and locations being
        2d numpy arrays.
        :param cutoffs: the maximum length of an edge in x and y,
        can be None for no maximum length
        :param find_perimeter: Whether or not to compute
        the 'infinite face' rings and store it in self.perimeter_rings
        """
        self.graph: Graph = graph
        self.remove_self_edges()
        self.coords_dict: Dict[Node, Coord] = copy.deepcopy(coords_dict)
        self.missing_policy = missing_policy

        # Tidying up stage -- remove the long edges,
        # and remove the single coordinate sites.
        self.cutoffs: np.array = cutoffs
        if cutoffs is not None:
            self.remove_long_edges()
        self.removed_nodes, self.removed_edges = self.remove_single_coordinate_sites()
        self.removable_edges = None
        self.perimeter_rings = None
        # Now triangulate the graph and do the real heavy lifting.
        self.tri_graph, self.simplices = self.triangulate_graph()
        self.current_rings = {
            Shape(node_list_to_edges(simplex), self.coords_dict)
            for simplex in self.simplices
        }
        self.identify_rings()
        # In the case of disjoint rings, there can be multiple perimeters.

        if find_perimeter:
            self.perimeter_rings = self.find_perimeter_rings()

    def remove_self_edges(self):
        """
        Remove all edges that loop round on themselves.

        A self edge is one that is (n, n). This screws up the
        ring finder because it uses frozensets to test edges,
        so throw these out. They are rings of size... 1?
        """
        to_remove = set()
        for edge in self.graph.edges:
            if len(set(edge)) == 1:
                to_remove.add(edge)
        self.graph.remove_edges_from(to_remove)

    def find_perimeter_rings(self):
        """
        Locate the perimeter ring of this arrangement.

        The perimeter ring is also known as the 'infinite face'.
        Must be called after we've found the other shapes,
        as we use that information to identify the perimeter ring.
        :return perimeter_rings: a set of the perimeter rings
        """
        # Count all the edges that are only used in one shape.
        # That means they're at the edge, so we can mark them
        # as the perimeter ring.
        edge_use_count = Counter(
            [edge for shape in self.current_rings for edge in shape.edges]
        )
        single_use_edges = {key for key, count in edge_use_count.items() if count == 1}
        single_use_edges = frozenset(single_use_edges)

        # These are lines connecting two 'rings', and must be
        # passed upwards.
        zero_use_edges = {
            frozenset(edge)
            for edge in self.graph.edges
            if edge_use_count[frozenset(edge)] == 0
        }
        zero_use_edges = frozenset(zero_use_edges)
        # Turn this list of edges into a graph and
        # count how many rings are in it.
        perimeter_ring_graph = nx.Graph()
        perimeter_ring_graph.add_edges_from(single_use_edges)
        perimeter_ring_graph.add_edges_from(zero_use_edges)
        perimeter_coords = {
            node: self.coords_dict[node] for node in perimeter_ring_graph.nodes()
        }
        sub_ring_finder = RingFinder(
            perimeter_ring_graph,
            coords_dict=perimeter_coords,
            cutoffs=None,
            find_perimeter=False,
            missing_policy=self.missing_policy,
        )
        if zero_use_edges:
            edge_rings = sub_ring_finder.current_rings.union({Shape(zero_use_edges)})
        else:
            edge_rings = sub_ring_finder.current_rings
        return edge_rings

    def remove_long_edges(self):
        """
        Remove any edges that are longer than a set of cutoffs.
        This is useful to make a periodic cell aperiodic.
        :return graph: a graph minus the edges that are too long.
        Note that this mutates the original graph, so the return value can be ignored.
        """
        to_remove = set()
        for edge in self.graph.edges():
            pos_a = self.coords_dict[edge[0]]
            pos_b = self.coords_dict[edge[1]]
            distance = np.abs(pos_b - pos_a)

            if np.any(distance > self.cutoffs):
                to_remove.add(edge)
        self.graph.remove_edges_from(to_remove)
        return self.graph

    def triangulate_graph(self):
        """
        Constructs a Delauney triangulation
        of a set of coordinates, and returns
        it as a networkx graph.
        :return tri_graph: a Delaunay triangulation of the original graph.
        :return mapped_simplices: a list of all the edges making up triangular simplicies
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
        try:
            delaunay_res = Delaunay(coords_array)
        except ValueError as ex:
            raise RingFinderError(str(ex))
        except RuntimeError as ex:
            raise RingFinderError(str(ex))
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
        :return graph: a graph minus the single coordinate notes. Note that this mutates the original graph, so the return value can be ignored.
        """
        removed_nodes = set()
        removed_edges = set()
        while True:
            # Find the 0 or 1 coordinate nodes and make a list of them,
            # then remove both their entry in the graph and their
            # coordinate.
            nodes_to_remove = [item[0] for item in self.graph.degree() if item[1] < 2]
            removed_nodes.update(nodes_to_remove)
            removed_edges.update(
                [
                    edge
                    for node in nodes_to_remove
                    for edge in list(self.graph.edges(node))
                ]
            )
            if not nodes_to_remove:
                break
            self.graph.remove_nodes_from(nodes_to_remove)
            for node in nodes_to_remove:
                del self.coords_dict[node]
        return removed_nodes, removed_edges

    def flip_degenerate_edge(self, edge) -> bool:
        """
        Flips a degenerate edge in a Delaunay triangulation
        in an attempt to match the original graph better.
        | \ | <-> | \ |
        Works by identifying if the edge is part of a rectangle,
        and removing this edge from self.tri_graph if it
        is, and adding the other diagonal.
        :return did_flip: did we successfully flip the edge?
        """
        # TODO: Same O(n^2) problem here! Even worse because
        # n_triangles is so very very big. Could optimise this
        # by precalculating it.
        nodes = list(edge)
        if nodes[0] not in self.tri_graph or nodes[1] not in self.tri_graph:
            return False
        neighbors = [set(self.tri_graph.neighbors(node)) for node in nodes]
        other_edge = tuple(neighbors[0].intersection(neighbors[1]))
        if len(other_edge) != 2:
            # There are more than two common edges between these two nodes.
            # That means this isn't a valid triangulation! Bail out.
            return False
        if other_edge in self.tri_graph.edges and other_edge not in self.graph.edges:
            self.tri_graph.remove_edge(*other_edge)
            self.tri_graph.add_edge(*edge)
            # We also need to reconstruct the simplices before we go any further.
            to_remove = []
            for shape in self.current_rings:
                if frozenset(other_edge) in shape:
                    # Note that because we've overridden __hash__
                    # we must construct a new shape.
                    to_remove.append(shape)
            for shape in to_remove:
                self.current_rings.remove(shape)

            for other_node in other_edge:
                new_edges = frozenset(
                    [frozenset([node, other_node]) for node in edge] + [edge]
                )
                new_shape = Shape(new_edges, coords_dict=self.coords_dict)
                self.current_rings.add(new_shape)
            return True
        return False

    def draw_missing(self, main_edge_set, missing_edges):
        """
        Draw the edges that are missing from this triangulation
        """
        fig, ax = plt.subplots()
        nx.draw_networkx_edges(
            self.graph,
            pos=self.coords_dict,
            edgelist=[tuple(item) for item in main_edge_set],
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.tri_graph, pos=self.coords_dict, style="dotted", ax=ax
        )
        nx.draw_networkx_edges(
            self.graph,
            pos=self.coords_dict,
            edgelist=[tuple(item) for item in missing_edges],
            ax=ax,
            edge_color="red",
            width=1.5,
        )
        nodes_in_missing_edges = set()
        for edge in missing_edges:
            nodes_in_missing_edges.update(edge)

        nx.draw_networkx_labels(
            self.graph,
            pos=self.coords_dict,
            labels={n: f"{n}" for n in nodes_in_missing_edges},
        )
        if self.cutoffs is not None:
            ax.set_xlim(0, self.cutoffs[0] * 2.0)
            ax.set_ylim(0, self.cutoffs[1] * 2.0)
        fig.savefig("./missing_edges.pdf")
        plt.close(fig)

    def identify_rings(self, max_to_remove: int = None):
        """
        Removes the edges from a triangulated graph that do not exist
        in the original graph, identifying rings in the process.
        Start off with a set of simplices as the building blocks
        of rings.
        :param max_to_remove: the maximum number of edges to remove. Useful for making animations, but is None by default.
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
            # There is one case where this is salvagable, and that's
            # the case of degenerate triangulations (i.e. |\| vs |/|)
            # Try to spot those before bailing out.
            for edge in missing_edges:
                did_flip = self.flip_degenerate_edge(edge)
                if not did_flip:
                    # If we didn't flip that one, it's still missing
                    # so we needn't bother with the rest.
                    # self.draw_missing(main_edge_set, missing_edges)

                    if self.missing_policy == "raise":
                        missing_edge_str = [str(tuple(item)) for item in missing_edges]
                        raise RingFinderError(
                            "There are edges in the main graph that do "
                            + "not exist in the Delauney triangulation: "
                            + f"{missing_edge_str}. Is your periodic box "
                            + "the right size?"
                        )
                    elif self.missing_policy == "remove":
                        self.graph.remove_edge(*edge)
                    elif self.missing_policy == "add":
                        self.tri_graph.add_edge(*edge)
                    elif self.missing_policy == "ignore":
                        continue
                    elif self.missing_policy == "return":
                        # self.current_rings = None
                        return
                    else:
                        raise RuntimeError(
                            "bad missing policy -- must be raise, remove, add or ignore"
                        )
            # Get here only if we successfully flipped all the edges.
            # Update the tri_edge_set.
            tri_edge_set = {frozenset(edge) for edge in self.tri_graph.edges()}

        self.removable_edges: Set[Edge] = tri_edge_set.difference(main_edge_set)
        if not self.removable_edges:
            # No removeable edges, so bail out.
            return
        if max_to_remove is None:
            max_to_remove = len(self.removable_edges)

        # Remove each edge one by one. The max_to_remove parameter
        # will halt this process in its tracks, so you'll have to call
        # this function again or manually remove edges. Useful for
        # making animations.
        edges_removed: int = 1


        edge: Edge = self.removable_edges.pop()
        while self.removable_edges:
            edges_removed += 1
            self.remove_one_edge(edge)
            edge = self.removable_edges.pop()
            if edges_removed > max_to_remove:
                return
        self.remove_one_edge(edge)

    def remove_one_edge(self, edge: Edge):
        """
        Removes a single edge from the Delaunay triangulation graph
        that does not exist in the 'main' graph. Checks which shapes
        in self.current_rings this edge belongs to, and updates them.
        There should only be one or two rings that each edge belongs to.
        :param edge: a frozenset of two ints representing
        the edge we wish to remove.
        """
        shapes_with_edge: Sequence[Shape] = []
        # TODO: This is O(n^2) so gets bad
        # pretty quickly. Maybe I should store
        # a dict.
        for shape in self.current_rings:
            if edge in shape:
                shapes_with_edge.append(shape)
            if len(shapes_with_edge) == 2:
                break

        if len(shapes_with_edge) == 1:
            # It's only part of one shape.
            # Scrap it.
            # TODO: this might have to change for periodic.
            self.current_rings.remove(shapes_with_edge[0])
            return

        if len(shapes_with_edge) == 0:
            # This is a stranded edge. This means
            # something has gone horribly wrong
            # and we should bail out.
            return

        # Mutate the class current_rings set, by removing
        # the two rings we just merged and adding the new one.
        new_shape: Shape = shapes_with_edge[0].merge(shapes_with_edge[1], edge=edge)
        for shape in shapes_with_edge:
            self.current_rings.remove(shape)
        self.current_rings.add(new_shape)

    def quick_draw(self, filename):
        fig, ax = plt.subplots()
        nx.draw(self.graph, pos=self.coords_dict, ax=ax, node_size=5)
        colors = ["red", "blue", "green", "orange", "pink", "brown"]
        if self.perimeter_rings is not None:
            for i, perimeter_ring in enumerate(self.perimeter_rings):
                nx.draw_networkx_edges(
                    self.graph,
                    pos=self.coords_dict,
                    ax=ax,
                    edgelist=[tuple(edge) for edge in perimeter_ring.edges],
                    edge_color=colors[i],
                    width=5.0,
                )
        fig.savefig(filename, dpi=800)
        plt.close(fig)

    def ring_sizes(self) -> Sequence[int]:
        """
        Returns the sizes of the rings in this shape.
        :return sizes: a list of ring sizes.
        """
        return [len(ring) for ring in self.current_rings]

    def as_polygons(self) -> Sequence[Polygon]:
        """
        Returns a list of the current rings as matplotlib
        polygon objects for ease of plotting.
        :return polygons: a list of polygon objects.
        """
        return [ring.to_polygon() for ring in self.current_rings]

    def draw_onto(
        self,
        ax,
        cmap_name: str = "viridis",
        color_by: str = "size",
        color_reversed:bool = False,
        min_ring_size=None,
        max_ring_size=None,
        **kwargs,
    ) -> None:
        """
        Draws the coloured polygons onto a matplotlib
        axis.
        """
        # Calculate the bounding boxes
        mins = np.array([np.inf, np.inf])
        maxes = np.array([-np.inf, -np.inf])
        for ring in self.current_rings:
            ring_bounding = ring.bounding_box()
            mins = np.minimum(mins, ring_bounding[:, 0])
            maxes = np.maximum(maxes, ring_bounding[:, 1])

        ax.set_xlim(mins[0], maxes[0])
        ax.set_ylim(mins[1], maxes[1])
        polys = self.as_polygons()

        if color_by == "size":
            color_data = self.ring_sizes()
        elif color_by == "regularity":
            color_data = [ring.regularity_metric() for ring in self.current_rings]
        # Sometimes we don't get the ring sizes right. The user can provide
        # a lower bound on the maximum ring size and an upper bound on the
        # minimum ring size for more consistent colouring.
        normalised_data = colors.Normalize(vmin=min_ring_size if min_ring_size is not None else min(color_data),
                                        vmax=max_ring_size if max_ring_size is not None else max(color_data),
                                        clip=True)(color_data)
        if color_reversed:
            normalised_data = np.ones_like(normalised_data) - normalised_data
        color_data = plt.cm.get_cmap(cmap_name)(normalised_data)

        p = PatchCollection(polys, linewidth=2.0)
        p.set_color(color_data)
        p.set_linestyle("dotted")
        p.set_edgecolor("black")
        ax.add_collection(p)
        edges_to_draw = [
            tuple(edge) for ring in self.current_rings for edge in ring.edges
        ]
        try:
            graph_to_plot = self.aperiodic_graph
        except AttributeError:
            graph_to_plot = self.graph
        nx.draw_networkx_edges(
            graph_to_plot,
            ax=ax,
            pos=self.coords_dict,
            edge_color="black",
            #zorder=1000,
            width=2.5,
            edgelist=edges_to_draw,
            **kwargs,
        )
        nx.draw_networkx_nodes(
            graph_to_plot,
            ax=ax,
            pos=self.coords_dict,
            node_color="black",
            node_size=2.5,
        )
        
    def to_tikz(self, filename: str, cmap: str="coolwarm", vmin=None, vmax=None):
        color_data = self.ring_sizes()
        if vmin is None:
            vmin = min(color_data)
        if vmax is None:
            vmax = max(color_data)
        color_lut = int(vmax - vmin)
        colors = cm.get_cmap(cmap)(np.linspace(0, 1, color_lut))
        with open(filename, "w") as fi:
            fi.write(r"\begin{tikzpicture}" + "\n")
            for idx, color in enumerate(colors):
                fi.write("\definecolor{" + f"{cmap}{vmax-vmin}v{i}" "}{RGB}{" + f"{int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}" + "}\n")
            
            for ring in self.current_rings:
                color_idx = min(vmax - vmin, len(ring) - vmin)
                
                for node in ring.to_node_list:
                    fi.write(r"\draw [thick, black, fill=" + f"{cmap}{vmax-vmin}v{i}] ")
                    for node in ring.to_node_list():
                        pos = pos_dict[node]
                        fi.write(f"({pos[0]:.2f}, {pos[1]:.2f}) -- ")
                    fi.write("cycle;\n")
            
            for u, v in self.graph.edges:
                fi.write(r"\draw[thick, black] " +f"({self.coords_dict[u][0]}, {self.coords_dict[u][1]}) -- ({self.coords_dict[v][0], self.coords_dict[v][1]})\n"
            
            for node in self.graph.nodes:
                fi.write(r"\node [circle,inner sep=0pt, fill=brewer1,  minimum size=10pt] " + f"(node{node}) at ({self.coords_dict[node][0]}, {self.coords_dict[node][1]}) {};")
            fi.write(r"\end{tikzpicture}")
            
           
            
    def analyse_edges(self):
        """
        Return a list of all of the edge lengths in the graph.

        """
        edge_lengths = []
        for u, v in self.graph.edges:
            gradient = self.coords_dict[v] - self.coords_dict[u]

            # If we're in a periodic box, we have to apply the
            # minimum image convention. Do this by creating
            # a virtual position for v, which is a box length away.
            # We need the += and -= to cope with cases where we're out in
            # both x and y.
            new_pos_v = self.coords_dict[v]
            if gradient[0] > self.cutoffs[0]:
                new_pos_v -= np.array([2 * self.cutoffs[0], 0.0])
            elif gradient[0] < -self.cutoffs[0]:
                new_pos_v += np.array([2 * self.cutoffs[0], 0.0])

            if gradient[1] > self.cutoffs[1]:
                new_pos_v -= np.array([0, 2 * self.cutoffs[1]])
            elif gradient[1] < -self.cutoffs[1]:
                new_pos_v += np.array([0, 2 * self.cutoffs[1]])
            new_gradient = new_pos_v - self.coords_dict[u]
            edge_lengths.append(np.hypot(*new_gradient))
        return edge_lengths


def convert_to_ring_graph(input_rings: Set[Shape]) -> nx.Graph:
    """
    Convert a set of rings into a 'ring graph', with nodes
    being rings and edges being shared edges between rings.
    :param input rings: an iterable of  rings
    """
    ring_graph = nx.Graph()
    ring_sizes = dict()
    ring_centres = dict()
    input_rings = list(input_rings)
    for i, ring in enumerate(input_rings):
        ring_sizes[i] = len(ring)
        ring_centres[i] = ring.centroid()
        for j in range(i):
            other_ring = input_rings[j]
            if ring.shared_edges(other_ring, 200) != 0:
                ring_graph.add_edge(i, j)

    nx.set_node_attributes(ring_graph, ring_sizes, "size")
    nx.set_node_attributes(ring_graph, ring_centres, "pos")
    return ring_graph


def topological_rdf(ring_graph: nx.Graph, compute_standard_error=True):
    """
    Calculate a topological RDF.

    A topological RDF is the average size of ring around a ring of
    size M, with distance being the number of shared edges away.

    :param ring_graph: DESCRIPTION
    :type ring_graph: nx.Graph
    :return: DESCRIPTION
    :rtype: TYPE
    """

    ring_sizes = nx.get_node_attributes(ring_graph, "size")
    if not ring_sizes:
        raise RuntimeError("Graph must have a ring size attribute.")

    ring_size_rdfs = dict()
    observed_ring_sizes = set()
    maximum_path = 0
    for node in ring_graph.nodes():
        shortest_paths = nx.single_source_shortest_path(ring_graph, source=node)
        this_node_size = ring_sizes[node]
        observed_ring_sizes.add(this_node_size)
        # initialise a blank dictionary
        if this_node_size not in ring_size_rdfs:
            ring_size_rdfs[this_node_size] = defaultdict(list)

        for other_node, path in shortest_paths.items():
            # subtract one here because we don't count the last node
            path_length = len(path) - 1
            maximum_path = max(maximum_path, path_length)
            other_node_size = ring_sizes[other_node]
            ring_size_rdfs[this_node_size][path_length].append(other_node_size)
            observed_ring_sizes.add(other_node_size)
    # now average the ring sizes for the rdf
    mean_ring_rdfs = dict()
    std_ring_rdfs = dict()
    for ring_size in sorted(list(observed_ring_sizes)):
        mean_ring_rdfs[ring_size] = [np.nan for _ in range(maximum_path + 1)]
        std_ring_rdfs[ring_size] = [np.nan for _ in range(maximum_path + 1)]

        this_rdf = ring_size_rdfs[ring_size]
        for distance, ring_sizes in this_rdf.items():
            if distance == 0:
                continue
            array_ring_sizes = np.array(ring_sizes)
            mean_ring_rdfs[ring_size][distance] = np.mean(array_ring_sizes)
            if len(ring_sizes) > 1:
                std_ring_rdfs[ring_size][distance] = np.std(array_ring_sizes, ddof=1)
                if compute_standard_error:
                    std_ring_rdfs[ring_size][distance] /= np.sqrt(len(ring_sizes))

    return mean_ring_rdfs, std_ring_rdfs


def geometric_rdf(
    ring_graph: nx.Graph, compute_standard_error=True, num_bins=100, box=None
):
    """
    Calculate a geometric RDF.

    A geometric RDF is the average size of ring around a ring of
    size M, with distance being the the distance between their centroids.

    :param ring_graph: DESCRIPTION
    :type ring_graph: nx.Graph
    :return: DESCRIPTION
    :rtype: TYPE
    """
    bin_size = np.hypot(*box) / num_bins
    ring_sizes = nx.get_node_attributes(ring_graph, "size")
    if not ring_sizes:
        raise RuntimeError("Graph must have a ring size attribute.")

    positions = nx.get_node_attributes(ring_graph, "pos")
    if not positions:
        raise RuntimeError("Graph must have a pos attribute.")
    ring_size_rdfs = {i: [[] for _ in range(num_bins)] for i in range(21)}
    for node in ring_graph.nodes():
        this_node_pos = positions[node]
        if np.any(np.isnan(this_node_pos)):
            continue
        this_ring_size = ring_sizes[node]
        for other_node in ring_graph.nodes():
            if node == other_node:
                continue
            other_node_pos = positions[other_node]
            if np.any(np.isnan(other_node_pos)):
                continue
            other_ring_size = ring_sizes[other_node]
            distance = np.abs(other_node_pos - this_node_pos)
            if distance[0] > box[0] / 2:
                distance[0] -= box[0]
            elif distance[0] < -box[0] / 2:
                distance[0] += box[0]
            if distance[1] > box[1] / 2:
                distance[1] -= box[1]
            elif distance[1] < -box[1] / 2:
                distance[1] += box[1]
            displacement = np.hypot(*distance)
            bin_id = int(displacement // bin_size)
            # Make sure we don't try to add a displacement outside of the range of interest
            if bin_id > 0 and bin_id < num_bins:
                ring_size_rdfs[this_ring_size][bin_id].append(other_ring_size)

    # now average the ring sizes for the rdf
    for key, val in ring_size_rdfs.items():
        for i, sublist in enumerate(val):
            ring_size_rdfs[key][i] = np.mean(sublist)
    return ring_size_rdfs


if __name__ == "__main__":
    coordinates = pd.read_csv("coordinates.dat")
    
