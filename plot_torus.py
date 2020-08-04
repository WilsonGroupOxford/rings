#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:44:15 2019

@author: matthew-bailey
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

edges = []
with open("./data/coll_edges.dat", "r") as fi:
    for line in fi.readlines():
        if line.startswith("#"):
            continue
        x, y = [int(item) for item in line.split(", ")]
        edges.append((x, y))

COORDS_DICT = {}
with open("./data/coll_coords.dat", "r") as fi:
    for i, line in enumerate(fi.readlines()):
        if line.startswith("#"):
            continue
        line = line.split(", ")
        node_id, x, y = int(line[0]), float(line[1]), float(line[2])
        COORDS_DICT[node_id] = np.array([x, y])

with open("./data/coll_rings.dat", "r") as fi:
    rings = []
    for line in fi.readlines():
        rings.append([int(node) for node in line.split()])


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

COORDS_INDICIES = {key: i for i, key in enumerate(COORDS_DICT.keys())}
COORDS_ARR = np.vstack([pos for pos in COORDS_DICT.values()])
# Now map to the Torus.
# Shift the smallest to be at 0.
MIN_COORDS_ARR = np.min(COORDS_ARR, axis=0)
COORDS_ARR -= MIN_COORDS_ARR

MAX_COORDS_ARR = np.max(COORDS_ARR, axis=0)
COORDS_ARR /= MAX_COORDS_ARR
COORDS_ARR *= 2 * np.pi
THETAS = COORDS_ARR[:, 0]
PHIS = COORDS_ARR[:, 1]
BIG_R = 2
LITTLE_R = 1
XS = (BIG_R + LITTLE_R * np.cos(THETAS)) * np.cos(PHIS)
YS = (BIG_R + LITTLE_R * np.cos(THETAS)) * np.sin(PHIS)
ZS = LITTLE_R * np.sin(THETAS)
CARTESIAN_ARR = np.vstack([XS, YS, ZS]).T
ax.scatter(CARTESIAN_ARR[:, 0], CARTESIAN_ARR[:, 1], CARTESIAN_ARR[:, 2])

sizes = [len(ring) for ring in rings]
size_range = max(sizes) + 1 - min(sizes)
this_cmap = plt.cm.get_cmap("viridis")(np.linspace(0, 1, size_range))
colours = [this_cmap[size - 4] for size in sizes]
all_ring_verts = []
for ring in rings:
    ring_verts = [CARTESIAN_ARR[COORDS_INDICIES[node]] for node in ring]
    all_ring_verts.append(ring_verts)

pc = Poly3DCollection(all_ring_verts, color=colours)
pc.set_zsort("max")
ax.add_collection3d(pc, zs=CARTESIAN_ARR[:, 2])

for edge_a, edge_b in edges:
    index_a = COORDS_INDICIES[edge_a]
    index_b = COORDS_INDICIES[edge_b]
    ax.plot(
        [CARTESIAN_ARR[index_a, 0], CARTESIAN_ARR[index_b, 0]],
        [CARTESIAN_ARR[index_a, 1], CARTESIAN_ARR[index_b, 1]],
        [CARTESIAN_ARR[index_a, 2], CARTESIAN_ARR[index_b, 2]],
        linewidth=10,
        color="black",
    )
fig.show()

with open("./model.obj", "w") as objfi:
    objfi.write("# Vertex positions\n")
    for i, coord in enumerate(CARTESIAN_ARR):
        objfi.write(f"v {coord[0]:.5f} {coord[1]:.5f} {coord[2]:.5f}\n")

    objfi.write("\n# Texture Vertices\n")
    for size in range(min(sizes), max(sizes) + 1):
        fractional_size = (size - min(sizes)) / max(sizes)
        objfi.write(f"vt {fractional_size:.3f} {0.0:.3f}\n")

    objfi.write("\n# Ring Faces\n")
    for ring in rings:
        # We actually want to write out the Delaunay triangulation
        # of this ring. Do the triangulation in 2D first.
        ring_coords = np.vstack([COORDS_ARR[COORDS_INDICIES[node]] for node in ring])
        relative_size = len(ring) - min(sizes)
        objfi.write(
            "f "
            + " ".join(
                [f"{COORDS_INDICIES[node] + 1}/{relative_size}" for node in ring]
            )
            + "\n"
        )
