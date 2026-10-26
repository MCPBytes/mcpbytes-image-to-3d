"""Smooth vertex normals that ignore UV seams (NumPy only).

A textured glTF mesh splits a vertex wherever its UVs jump: the copies share a position but not a triangle fan, so
normals computed per index give each copy only its own side's triangles and every seam shades as a hard crease (a
finished TRELLIS model measured a median 19°, p90 48° error on its seam copies). Here the normal belongs to the
position: copies are welded, each welded point sums all of its triangles, and every copy gets that one normal back.
Generated surfaces are closed and carry no intended creases, so every split they have is a seam.
"""

import numpy as np

UP = (0.0, 1.0, 0.0)  # glTF is +Y up: a vertex with no surface around it faces up rather than nowhere
# Below this share of its triangles' area, a welded point's sum is float noise (a fan that folds back on itself), not
# a direction.
CANCELLED = 1e-9


def welded_vertex_normals(vertices, faces, tolerance=1e-6):
    """Unit (N, 3) normals for `vertices`, shared by every copy of a position.

    Positions weld on a grid of `tolerance` × the largest bounding-box extent (UV-seam copies are bit-identical, so
    they always land in the same cell). Area-weighted, not angle-weighted: the sum of the triangles' raw cross
    products is the area-weighted normal for free, and the near-degenerate slivers that decimation leaves (whose
    direction is float32 noise, and whose one obtuse corner would dominate an angle weighting) add almost nothing.
    It is also what three.js computes for a mesh without normals. Zero-area faces add nothing; a vertex with no area
    around it gets +Y.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(vertices) or not len(faces):
        return np.tile(np.array(UP), (len(vertices), 1))
    low = vertices.min(axis=0)
    extent = float((vertices.max(axis=0) - low).max())
    grid = np.round((vertices - low) / (tolerance * extent if extent > 0 else 1.0)).astype(np.int64)
    weld = np.unique(grid, axis=0, return_inverse=True)[1].reshape(-1)  # NumPy 2.0.0 returned it (N, 1)
    count = int(weld.max()) + 1
    raw = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])  # 2 × area along the normal
    corner = weld[faces].reshape(-1)
    summed = np.stack([np.bincount(corner, weights=np.repeat(raw[:, axis], 3), minlength=count) for axis in range(3)], axis=1)
    area = np.bincount(corner, weights=np.repeat(np.linalg.norm(raw, axis=1), 3), minlength=count)
    length = np.linalg.norm(summed, axis=1)
    usable = length > CANCELLED * area  # strict: no area at all (isolated, or only zero-area faces) fails too
    shared = np.tile(np.array(UP), (count, 1))
    shared[usable] = summed[usable] / length[usable, None]
    return shared[weld]
