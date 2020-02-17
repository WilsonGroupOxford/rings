#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:32:13 2019

@author: matthew-bailey
"""

import copy
from collections import Counter, defaultdict
from typing import Any, Dict, NewType, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PIL
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay

try:
    from .ring_finder import RingFinder
    from .shape import Shape, node_list_to_edges
except ImportError:
    from ring_finder import RingFinder
    from shape import Shape, node_list_to_edges

Node = NewType("Node", int)
Graph = NewType("Graph", nx.Graph)
Coord = NewType("Coord", np.array)


class PeriodicRingFinder(RingFinder):
    def __init__(self, graph: Graph, coords_dict: Dict[Node, Coord], cell=None):
        self.graph: Graph = copy.deepcopy(graph)

        periodic_graph = copy.deepcopy(graph)
        periodic_coords = copy.deepcopy(coords_dict)
        self.cell = cell
        self.cutoffs = cell / 2.0
        self.coords_dict: Dict[Node, Coord] = coords_dict
        self.original_nodes = {key for key in self.coords_dict.keys()}
        self.perimeter_rings = None
        # First, do the aperiodic computation.
        super().__init__(
            graph=self.graph,
            coords_dict=self.coords_dict,
            cutoffs=cell / 2.0,
            find_perimeter=True,
        )
        self.aperiodic_graph = self.graph
        self.graph = periodic_graph
        self.coords_dict = periodic_coords
        # Tidying up stage -- remove the long edges,
        # and remove the single coordinate sites.
        self.add_periodic_images()
        self.add_periodic_edges()
        self.graph = self.remove_long_edges()
        self.remove_single_coordinate_sites()
        self.removable_edges = None
        # Now triangulate the graph and do the real heavy lifting.
        self.tri_graph, self.simplices = self.triangulate_graph()
        self.current_rings = {
            Shape(node_list_to_edges(simplex), self.coords_dict)
            for simplex in self.simplices
        }

        self.identify_rings()
        self.current_rings = self.find_unique_rings()

    def find_unique_rings(self):
        """
        Each ring in self.current_shapes has 8 periodic images.
        Identify just one of each, and keep that.
        """
        num_nodes = len(self.original_nodes)
        unique_rings = defaultdict(set)
        for ring in self.current_rings:
            modulo_edges = set()
            for edge in ring.edges:
                modulo_edge = frozenset({item % num_nodes for item in edge})
                modulo_edges.add(modulo_edge)
            new_ring = Shape(modulo_edges)
            unique_rings[new_ring].add(ring)

        # The "perimeter ring" is misidentified as a consequence
        # of the images we use. Thankfully, we know that
        # other rings can appear 1x, 2x, 4x, 6x or 9x.
        # Thus, a ring that appears 8x is always spurious.
        to_remove = set()
        for ring, ring_copies in unique_rings.items():
            if len(ring_copies) == 8:
                to_remove.add(ring)
        if to_remove:
            for ring in to_remove:
                del unique_rings[ring]
        # Now, from this set of unique rings we pick just
        # one to plot -- the one that has the most nodes
        # in the original periodic image.
        canonical_rings = set()
        for unique_ring, copy_rings in unique_rings.items():
            copy_rings = list(copy_rings)
            num_shared_edges = []
            num_shared_nodes = []
            nodes_unique = {item for edge in unique_ring.edges for item in edge}
            for copy_ring in copy_rings:
                shared_edges = unique_ring.edges.intersection(copy_ring.edges)
                nodes_copy = {item for edge in copy_ring.edges for item in edge}
                shared_nodes = nodes_unique.intersection(nodes_copy)

                num_shared_edges.append(len(shared_edges))
                num_shared_nodes.append(len(shared_nodes))
            max_shared_edges = max(num_shared_edges)
            # If none of the mirror rings share a node with
            # the 'original ring', it means that the original
            # ring doesn't really exist. It's spurious.
            # I can probably rectify that some other way,
            # but throwing it out seems the best away for now.
            if max(num_shared_nodes) == 0:
                continue
            max_shared_idx = num_shared_edges.index(max_shared_edges)
            canonical_rings.add(copy_rings[max_shared_idx])
        return canonical_rings

    def add_link_between(self, node_a, node_b, image_a, image_b):
        """
        Adds edges between all combinations of node_a and node_b
        across two periodic images (i.e. (a_a, b_b), (a_b, b_a)
        (a_a, b_a), (a_b, b_b)). Then we let remove_long_edges
        sort them out.
        :param node_a:
        :param node_b:
        :param image_a: a tuple in the form (x, y) indicating the offset of this image from the centre
        :param image_b: a tuple in the form (x, y) indicating the offset of this image from the centre.
        """

        # Make sure (0, 0) is the first offset, so we don't duplicate
        # the central nodes. The rest can be in any order as we
        # look them up later.
        num_nodes = len(self.original_nodes)
        cell_offsets = [
            (0, 0),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]
        assert image_a in cell_offsets, f"{image_a} not in {cell_offsets}"
        assert image_b in cell_offsets, f"{image_b} not in {cell_offsets}"
        assert (
            0 <= node_a and node_a <= num_nodes
        ), f"{node_a} must be a non-image index (0 <= node_a <= {num_nodes})"
        assert (
            0 <= node_b and node_b <= num_nodes
        ), f"{node_b} must be a non-image index (0 <= node_b <= {num_nodes})"

        image_a_offset = num_nodes * cell_offsets.index(image_a)
        image_b_offset = num_nodes * cell_offsets.index(image_b)
        self.graph.add_edge(node_a + image_a_offset, node_b + image_b_offset)
        self.graph.add_edge(node_a + image_b_offset, node_b + image_a_offset)

    def add_periodic_images(self):
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
        cell_offsets = [
            (0, 0),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]

        to_add = set()
        perimeter_nodes = set()
        for ring in self.perimeter_rings:
            perimeter_nodes.update(ring.nodes)
        perimeter_nodes.update(self.removed_nodes)
        edge_images = set()
        for node in perimeter_nodes:
            edge_images.update(
                frozenset([node, item]) for item in set(self.graph.neighbors(node))
            )

        for edge_a, edge_b in edge_images:
            a_pos = self.coords_dict[edge_a]
            b_pos = self.coords_dict[edge_b]
            for i, cell_offset in enumerate(cell_offsets):
                offset = np.array(cell_offset) * self.cell
                new_edge_a = (i * num_nodes) + edge_a
                new_edge_b = (i * num_nodes) + edge_b
                new_a_pos = a_pos + offset
                new_b_pos = b_pos + offset
                periodic_coords_dict[new_edge_a] = new_a_pos
                periodic_coords_dict[new_edge_b] = new_b_pos
                to_add.add((new_edge_a, new_edge_b))

        self.graph.add_edges_from(to_add)
        self.coords_dict.update(periodic_coords_dict)

    def add_periodic_edges(self):
        """
        Turns periodic edges into minimum-image convention
        edges. Finds the edges that are longer than
        half a unit cell, and turns them into edges
        between neighbouring images.
        TODO:  remove abhorrent kafkaesque arithmetic 2019-11-13
        """
        perimeter_nodes = set()
        for ring in self.perimeter_rings:
            perimeter_nodes.update(ring.nodes)
        perimeter_nodes.update(self.removed_nodes)

        edge_images = set()
        for node in perimeter_nodes:
            edge_images.update(
                frozenset([node, item]) for item in set(self.graph.neighbors(node))
            )
        handled_nodes = set(self.aperiodic_graph.nodes())
        handled_nodes = handled_nodes.difference(perimeter_nodes)
        num_nodes = len(self.original_nodes)
        for edge in edge_images:
            edge = tuple(edge)
            if (
                edge[0] % num_nodes in handled_nodes
                and edge[1] % num_nodes in handled_nodes
            ):
                continue
            pos_a = self.coords_dict[edge[0]]
            pos_b = self.coords_dict[edge[1]]
            distance = np.abs(pos_b - pos_a)
            if distance[0] > self.cutoffs[0] and distance[1] > self.cutoffs[1]:
                if np.all(pos_a < pos_b) or np.all(pos_a > pos_b):
                    # This is a bottom-left top-right link. Add in a
                    # link from (0, 0) to (1, 1) and (0, 0) to (-1, -1).
                    self.add_link_between(edge[0], edge[1], (0, 0), (1, 1))
                    self.add_link_between(edge[0], edge[1], (0, 0), (-1, -1))
                    self.add_link_between(edge[0], edge[1], (-1, 0), (0, 1))
                    self.add_link_between(edge[0], edge[1], (0, -1), (1, 0))
                else:
                    # This is a top-left bottom-right link. Add in a
                    # link from (0, 0) to (-1, 1) and (0, 0) to (-1, 1).
                    self.add_link_between(edge[0], edge[1], (0, 0), (-1, 1))
                    self.add_link_between(edge[0], edge[1], (0, 0), (1, -1))
                    self.add_link_between(edge[0], edge[1], (1, 0), (0, 1))
                    self.add_link_between(edge[0], edge[1], (0, -1), (-1, 0))
            elif distance[1] > self.cutoffs[1]:
                # There is an edge that spans the y-coordinate.
                # Remove it, and add in the six new edges
                self.add_link_between(edge[0], edge[1], (-1, 0), (-1, -1))
                self.add_link_between(edge[0], edge[1], (0, 0), (0, -1))
                self.add_link_between(edge[0], edge[1], (1, 0), (1, -1))

                self.add_link_between(edge[0], edge[1], (-1, 1), (-1, 0))
                self.add_link_between(edge[0], edge[1], (0, 1), (0, 0))
                self.add_link_between(edge[0], edge[1], (1, 1), (1, 0))

            elif distance[0] > self.cutoffs[0]:
                # There is an edge that spans the x-coordinate.
                # Remove it, and add in the six new edges.
                self.add_link_between(edge[0], edge[1], (1, -1), (0, -1))
                self.add_link_between(edge[0], edge[1], (1, 0), (0, 0))
                self.add_link_between(edge[0], edge[1], (1, 1), (0, 1))

                self.add_link_between(edge[0], edge[1], (0, -1), (-1, -1))
                self.add_link_between(edge[0], edge[1], (0, 0), (-1, 0))
                self.add_link_between(edge[0], edge[1], (0, 1), (-1, 1))


if __name__ == "__main__":
    G: Graph = nx.Graph()
    with open("./data/coll_edges.dat", "r") as fi:
        fi.readline()  # Skip header
        for line in fi.readlines():
            x, y = [int(item) for item in line.split(",")]
            G.add_edge(x, y)

    COORDS_DICT: Dict[Node, Coord] = {}
    with open("./data/coll_coords.dat", "r") as fi:
        fi.readline()  # Skip header
        for line in fi.readlines():
            line = line.split(",")
            node_id, x, y = int(line[0]), float(line[1]), float(line[2])
            COORDS_DICT[node_id] = np.array([x, y])
    XS = [item[0] for item in COORDS_DICT.values()]
    YS = [item[1] for item in COORDS_DICT.values()]
    ring_finder = PeriodicRingFinder(
        G, COORDS_DICT, np.array([max(XS) - min(XS), max(YS) - min(YS)])
    )

    FIG, AX = plt.subplots()
    FIG.patch.set_visible(False)
    AX.axis("off")
    ring_finder.draw_onto(AX)
    AX.set_xlim(-95, 180)
    AX.set_ylim(-95, 180)
    FIG.savefig("./periodic-graph.pdf")
