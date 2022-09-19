#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:44:15 2019

@author: matthew-bailey
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
from periodic_ring_finder import PeriodicRingFinder
from shape import node_list_to_edges
import subprocess
from scipy.spatial import Delaunay

def hexagonal_lattice_graph(
    m: int,
    n: int,
    periodic: bool = False,
    with_positions: bool = True,
    create_using: nx.Graph = None,
) -> nx.Graph:
    """
    Return an `m` by `n` hexagonal lattice graph.

    Taken from networkx source code and modified by me, to remove the offset.

    The *hexagonal lattice graph* is a graph whose nodes and edges are
    the `hexagonal tiling`_ of the plane.

    The returned graph will have `m` rows and `n` columns of hexagons.
    `Odd numbered columns`_ are shifted up relative to even numbered columns.

    Positions of nodes are computed by default or `with_positions is True`.
    Node positions creating the standard embedding in the plane
    with sidelength 1 and are stored in the node attribute 'pos'.
    `pos = nx.get_node_attributes(G, 'pos')` creates a dict ready for drawing.

    .. _hexagonal tiling: https://en.wikipedia.org/wiki/Hexagonal_tiling
    .. _Odd numbered columns: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/

    Parameters
    ----------
    : param m : The number of rows of hexagons in the lattice.

    :param n : The number of columns of hexagons in the lattice.

    : periodic : Whether to make a periodic grid by joining the boundary vertices.
        For this to work `n` must be odd and both `n > 1` and `m > 1`.
        The periodic connections create another row and column of hexagons
        so these graphs have fewer nodes as boundary nodes are identified.

    : with_positions : (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with vertical columns of hexagons
        offset to interleave and cover the plane.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    :param create_using : NetworkX graph
        If specified, this must be an instance of a NetworkX graph
        class. It will be cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
        If graph is directed, edges will point up or right.

    :return: The *m* by *n* hexagonal lattice graph.
    """
    G = create_using if create_using is not None else nx.Graph()
    G.clear()
    if m == 0 or n == 0:
        return G
    if periodic and (n % 2 == 1 or m < 2 or n < 2):
        msg = "periodic hexagonal lattice needs m > 1, n > 1 and even n"
        raise nx.NetworkXError(msg)

    M = 2 * m  # twice as many nodes as hexagons vertically
    rows = range(M + 2)
    cols = range(n + 1)
    # make lattice
    col_edges = (((i, j), (i, j + 1)) for i in cols for j in rows[: M + 1])
    row_edges = (((i, j), (i + 1, j)) for i in cols[:n] for j in rows if i % 2 == j % 2)
    G.add_edges_from(col_edges)
    G.add_edges_from(row_edges)
    # Remove corner nodes with one edge
    G.remove_node((0, M + 1))
    G.remove_node((n, (M + 1) * (n % 2)))

    # identify boundary nodes if periodic
    if periodic:
        for i in cols[:n]:
            G = nx.contracted_nodes(G, (i, 0), (i, M))
        for i in cols[1:]:
            G = nx.contracted_nodes(G, (i, 1), (i, M + 1))
        for j in rows[1:M]:
            G = nx.contracted_nodes(G, (0, j), (n, j))
        G.remove_node((n, M))

    # calc position in embedded space
    ii = (i for i in cols for j in rows)
    jj = (j for i in cols for j in rows)
    xx = (0.5 + i + i // 2 + (j % 2) * ((i % 2) - 0.5) for i in cols for j in rows)
    h = np.sqrt(3) / 2
    yy = (h * j for i in cols for j in rows)
    # exclude nodes not in G
    pos = {
        (i, j): np.array([x, y]) for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in G
    }
    nx.set_node_attributes(G, pos, "pos")
    return G


def convert_to_torus(graph, periodic_box, inner_r=1.0, outer_r=2.0):
    pos_dict = nx.get_node_attributes(graph, "pos")
    coords_arr = np.vstack([pos for pos in pos_dict.values()])
    print(np.min(periodic_box, axis=1))
    cartesian_arr = project_to_torus(coords_arr, periodic_box, inner_r, outer_r)

    torus_dict = {key: cartesian_arr[key, :] for key in pos_dict.keys()}
    nx.set_node_attributes(graph, torus_dict, "torus_pos")
    return graph

def project_to_torus(arr, periodic_box, inner_r, outer_r):

    arr -= np.min(periodic_box, axis=1)
    arr /= np.max(periodic_box, axis=1) - np.min(periodic_box, axis=1)
    arr *= 2 * np.pi

    thetas, phis = arr[:, 0], arr[:, 1]

    xs = (outer_r + (inner_r * np.cos(thetas))) * np.cos(phis)
    ys = (outer_r + (inner_r * np.cos(thetas))) * np.sin(phis)
    zs = inner_r * np.sin(thetas)

    cartesian_arr = np.vstack([xs, ys, zs]).T
    return cartesian_arr


def scatter_vertices(graph, ax):
    torus_arr = np.vstack([pos for pos in nx.get_node_attributes(graph, "torus_pos").values()])
    ax.scatter(torus_arr[:, 0], torus_arr[:, 1], torus_arr[:, 2])


def draw_edges(graph, ax):
    torus_dict = nx.get_node_attributes(graph, "torus_pos")
    for u, v in graph.edges():
        pos_u, pos_v = torus_dict[u], torus_dict[v]
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]])
    return ax


def draw_faces(graph, rings, ax):
    torus_arr = np.vstack([pos for pos in nx.get_node_attributes(graph, "torus_pos").values()])
    sizes = [len(ring) for ring in rings]

    all_ring_verts = []
    for ring in rings:
        ring_verts = np.vstack([torus_arr[node] for node in ring.to_node_list()])
        all_ring_verts.append(ring_verts)

    pc = Poly3DCollection(all_ring_verts)
    pc.set_array(sizes)
    pc.set_zsort("max")
    ax.add_collection3d(pc, zs=torus_arr[:, 2])


def write_obj(graph, rings, filename="./model.obj"):
    torus_arr = np.vstack([pos for pos in nx.get_node_attributes(graph, "torus_pos").values()])
    with open(filename, "w") as objfi:
        objfi.write("# Vertex positions\n")
        for coord in torus_arr:
            objfi.write(f"v {coord[0]:.5f} {coord[1]:.5f} {coord[2]:.5f}\n")

        sizes = [ring.sizes for ring in rings]
        for size in range(min(sizes), max(sizes) + 1):
            fractional_size = (size - min(sizes)) / max(sizes)
            objfi.write(f"vt {fractional_size:.3f} {0.0:.3f}\n")

        objfi.write("\n# Ring Faces\n")
        for ring in rings:

            relative_size = len(ring) - min(sizes)
            objfi.write(
                "f "
                + " ".join(
                    [f"{node + 1}/{relative_size}" for node in ring]
                )
                + "\n"
            )

def triangulate_graph(graph):
    coords_dict = nx.get_node_attributes(graph, "pos")
    coords_array = np.empty([len(coords_dict), 2])
    index_to_key = {}
    for i, key in enumerate(sorted(coords_dict.keys())):
        index_to_key[i] = key
        coords_array[i, :] = coords_dict[key]

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


def write_tikz(graph, rings, periodic_box, filename="./torus.tex"):
    theta, phi = 0.0, 45.0
    cam_theta, cam_phi = np.radians(theta-90.0), np.radians(phi)
    cam_dir = np.array([np.sin(cam_phi) * np.cos(cam_theta),
                        np.sin(cam_phi) * np.sin(cam_theta),
                        np.cos(cam_phi)])

    light_theta, light_phi = np.radians(45.0), np.radians(90.0)
    light_direction = np.array([np.sin(light_phi) * np.cos(light_theta),
                                np.sin(light_phi) * np.sin(light_theta),
                                np.cos(light_theta)])

    with open(filename, "w") as fi:
        fi.write(r"\tdplotsetmaincoords{" + f"{int(phi)}" + r"}{" + f"{int(theta)}" + "}\n")
        fi.write(r"\begin{tikzpicture}[tdplot_main_coords]" + "\n")

        torus_dict = nx.get_node_attributes(graph, "torus_pos")
        pos_dict = nx.get_node_attributes(graph, "pos")
        #for u, v in graph.edges:
        #    fi.write("\t" + r"\draw[thick, black] " + f"({torus_dict[u][0]:.2f}, {torus_dict[u][1]:.2f}, {torus_dict[u][2]:.2f})" + " -- " + f"({torus_dict[v][0]:.2f}, {torus_dict[v][1]:.2f}, {torus_dict[v][2]:.2f})"  + ";\n")

        for ring in rings:
            node_list = ring.to_node_list()
            ring_coords = np.vstack([pos_dict[node] for node in node_list])
            delaunay = Delaunay(ring_coords)

            mean_normal = ring.normal_vector(embedding=torus_dict)
            ring_dp = np.dot(mean_normal, cam_dir)
            if ring_dp <= 0:
                continue

            for simplex in delaunay.simplices:
                triangle_coords = ring_coords[simplex]
                tor_triangle = np.vstack([torus_dict[node_list[node]]
                                          for node in simplex])
                vec_a = tor_triangle[0, :] - tor_triangle[1, :]
                vec_b = tor_triangle[2, :] - tor_triangle[1, :]
                normal_vec = np.cross(vec_a, vec_b)
                normal_vec /= np.linalg.norm(normal_vec)


                cam_dp = np.dot(normal_vec, cam_dir)
                if cam_dp < 0 and ring_dp < 0:
                    continue

                dp = np.dot(normal_vec, light_direction)
                color = max(0.0, min(int( (1 - dp) * 100), 90.0))

                fi.write(r"\draw[fill=" +
                         f"brewer1!{color}, brewer1!{color}] ")
                for idx, node in enumerate(simplex):
                    pos = tor_triangle[idx, :]
                    fi.write(f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) -- ")
                fi.write("cycle;\n")


            fi.write(r"\draw [thick, black] ")
            for node in ring.to_node_list():
                pos = torus_dict[node]
                fi.write(f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) -- ")
            fi.write("cycle;\n")
        fi.write(r"\end{tikzpicture}")

def write_tikz_planar(graph, rings, filename="./planar-hex.tex"):
    pos_dict = nx.get_node_attributes(graph, "pos")
    with open(filename, "w") as fi:
        fi.write(r"\begin{tikzpicture}" + "\n")
        for ring in rings:
            fi.write(r"\draw [thick, black, fill=brewer1] ")
            for node in ring.to_node_list():
                pos = pos_dict[node]
                fi.write(f"({pos[0]:.2f}, {pos[1]:.2f}) -- ")
            fi.write("cycle;\n")
        fi.write(r"\end{tikzpicture}" + "\n")
def main():
    num_nodes = 16
    G = hexagonal_lattice_graph(num_nodes, num_nodes)
    G = nx.convert_node_labels_to_integers(G)
    periodic_box = np.array(
        [
            [0.0, 1.5 * num_nodes+0.01],
            [0.0, num_nodes * np.sqrt(3)+0.01],
            #[-0.5 * num_nodes * np.sqrt(3), 0.5 * num_nodes * np.sqrt(3)],
        ]
    )

    G = convert_to_torus(G, periodic_box)

    # fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")

    #scatter_vertices(G, ax)
    #draw_edges(G, ax)

    rf = PeriodicRingFinder(G, nx.get_node_attributes(G, "pos"), periodic_box)
    fig, ax = plt.subplots()
    rf.draw_onto(ax)
    ax.axis("off")

    # draw_faces(G, rf.current_rings, ax)
    write_tikz(G, rf.current_rings, periodic_box)
    write_tikz_planar(G, rf.current_rings)
    #plt.show()
    #subprocess.run(["pdflatex", "test_torus.tex"], capture_output=False, shell=False, stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    main()


