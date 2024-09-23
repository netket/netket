import numpy as np
from itertools import product

from ._lattice_edge_logic import get_naive_edges, create_padded_sites


# Function to get N distinct colors
def get_n_colors(N):
    import matplotlib.pyplot as plt

    # Use 'tab10' colormap for distinct colors
    colors = plt.get_cmap("tab10", N)
    return [colors(i) for i in range(N)]


def default_arguments(dim, node_size, node_text_color, node_text_offset):
    """
    Compute default arguments when unspecified
    """
    if node_size is None:
        if dim == 1 or dim == 2:
            node_size = 300
        else:
            node_size = 50
    if node_text_offset is None:
        if dim == 1:
            node_text_offset = 0
        elif dim == 2:
            node_text_offset = 0
        elif dim == 3:
            node_text_offset = 0.05
    if node_text_color is None:
        if node_text_offset == 0:
            node_text_color = "white"
        else:
            node_text_color = "black"

    if isinstance(node_text_offset, (float, int)):
        node_text_offset = np.full((dim,), node_text_offset)
    return node_size, node_text_color, node_text_offset


def ax_text_autodim(
    ax,
    position,
    label,
    *,
    offset: list | float = 0,
    **kwargs,
):
    """
    Equivalent to ax.text but automatically handles
    1D/2D/3D positions, with an optional offset.
    """
    dim = len(position)
    if dim < 1 or dim > 3:
        raise NotImplementedError("...")

    if isinstance(offset, (float, int)):
        offset = np.full((dim,), 0.0)
    if isinstance(offset, (list, tuple)):
        offset = np.array(offset)

    if dim == 1:
        # 1D is still a 2D plot, the offset is along y
        dpos = np.array([position[0], offset[0]])
    else:
        dpos = position + offset

    return ax.text(*dpos, label, **kwargs)


def ax_scatter_autodim(
    ax,
    positions,
    *args,
    **kwargs,
):
    """
    Equivalent to ax.text but automatically handles
    1D/2D/3D positions, with an optional offset.
    """
    dim = positions.shape[-1]
    if dim < 1 or dim > 3:
        raise NotImplementedError("...")

    if positions.ndim == 1:
        positions = np.atleast_2d(positions)

    if dim == 1:
        # 1D is still a 2D plot, the offset is along y
        pos = [positions[:, 0], np.zeros_like(positions[:, 0])]
    elif dim == 2:
        pos = [positions[:, 0], positions[:, 1]]
    elif dim == 3:
        pos = [positions[:, 0], positions[:, 1], positions[:, 2]]

    return ax.scatter(
        *pos,
        *args,
        **kwargs,
    )


def draw_lattice(
    self,
    ax=None,
    figsize: tuple[int | float] | None = None,
    distance_order=1,
    *,
    node_size: int | None = None,
    node_color: str = "#1f78b4",
    node_text_color: str | None = None,
    node_text_offset: float | None = None,
    extra_sites_alpha: float = 0.3,
    draw_neighbors: bool = True,
    draw_text: bool = True,
    draw_basis_vectors: bool = True,
    draw_unit_cell: bool = True,
    show: bool = True,
    **kwargs,
):
    """
    Draws the ``Lattice`` graph

    Args:
        ax: A Matplotlib axis object. If unspecified it creates a figure.
        figsize: (width, height) tuple of the generated figure, if no axes is
            specified.
        node_size: Size of the nodes (as in matplotlib.pyplot.scatter).
        node_color: String with the colour of the nodes.
        node_text_color: String with the colour of the integer labelling the nodes.
            Defaults to white for 1D/2D and black in 3D.
        node_text_offset: float or 1/2/3-vector of offset for the label from the node.
            Defaults to 0 for 1/2D and to 0.05 for 3D.
        extra_sites_alpha: A float value between 0 and 1 for the alpha of the sites
            outside of the extent. Defaults to 0.3
        extra_sites_alpha: If true, we also draw sites outside of the extent
            of the lattice corresponding to the periodic images. (Defaults True)
        draw_text: If True, we draw the integer label with the site index (defaults to True).
        draw_basis_vectors: If True, draw the basis vectors (defaults to True).
        draw_unit_cell: If True, shade an area corresponding to the unit cell starting
            at the origin (Defaults to True).
        show: If True, show the plot before returning it (defaults to True).

    Returns:
        Matplotlib axis object containing the graph's drawing.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lattice = self

    # Determine the dimensionality of the lattice
    basis_vectors = lattice.basis_vectors
    dim = basis_vectors.shape[0]

    if dim < 1 or dim > 3:
        raise TypeError("Cannot draw 0D or 4D lattices")

    if len(kwargs.keys()) > 0:
        for deprecated_key in ("edge_color", "curvature", "font_size", "font_color"):
            if deprecated_key in kwargs.keys():
                kwargs.pop(deprecated_key)
                print(
                    f"Keyword argument {deprecated_key} is deprecated and does nothing anymore."
                )
        if len(kwargs.keys()) > 0:
            raise NotImplementedError(f"unsupported kwargument {tuple(kwargs.keys())}")

    node_size, node_text_color, node_text_offset = default_arguments(
        dim, node_size, node_text_color, node_text_offset
    )

    # extent = lattice.extent
    distance_atol = 1e-5

    # Set up the plot
    if ax is None:
        if dim == 1 or dim == 2:
            fig, ax = plt.subplots(figsize=figsize)
        else:  # 3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

    # Calculate all site positions
    all_sites = []
    for cell in product(*[range(e) for e in lattice.extent]):
        for offset in lattice.site_offsets:
            position = np.dot(cell, basis_vectors) + offset
            all_sites.append(position)
    all_sites = np.array(all_sites)

    all_sites, ids = create_padded_sites(
        basis_vectors, lattice.extent, lattice.site_offsets, lattice.pbc, 0
    )
    if draw_neighbors:
        all_sites_extra, ids_extra = create_padded_sites(
            basis_vectors,
            lattice.extent,
            lattice.site_offsets,
            lattice.pbc,
            distance_order,
        )

        naive_edges_by_order = get_naive_edges(
            all_sites_extra,
            distance_order * np.linalg.norm(basis_vectors, axis=1).max()
            + distance_atol,
            distance_order,
        )

        # remove virtual vertices that are not connected to physical ones
        union_vertices_to_delete = []
        to_delete = set()
        to_keep = set()
        for k, naive_edges in enumerate(naive_edges_by_order):
            for node1, node2 in naive_edges:
                # switch to real node indices
                pos1 = all_sites_extra[node1]
                pos2 = all_sites_extra[node2]

                # check if not within lattice
                pos1_id = lattice.positions[ids_extra[node1]]
                pos2_id = lattice.positions[ids_extra[node2]]

                if not (all(pos1_id == pos1) or all(pos2_id == pos2)):
                    to_delete.add(node1)
                    to_delete.add(node2)
                    # print(f"should delete {node1}, {node2} ({ids_extra[node1]},{ids_extra[node2]}) : ")
                    # print(f" -> {pos1_id =}, {pos1 = }, {pos2_id = }, {pos2 = }")
                else:
                    to_keep.add(node1)
                    to_keep.add(node2)
            # union_vertices_to_delete.append(vertices_to_delete)
        # union_vertices_to_delete = set(*union_vertices_to_delete)
        to_delete.difference_update(to_keep)
        union_vertices_to_delete = np.array(list(to_delete), dtype=np.int32)

        # for i in sorted(union_vertices_to_delete, reverse=True):
        #   print("deleting i", all_sites_extra[i])
        #   del all_sites_extra[i]
        #   del ids_extra[i]
        all_sites_extra = np.delete(all_sites_extra, union_vertices_to_delete, axis=0)
        ids_extra = np.delete(ids_extra, union_vertices_to_delete)

        # rgenerea
        naive_edges_by_order = get_naive_edges(
            all_sites_extra,
            distance_order * np.linalg.norm(basis_vectors, axis=1).max()
            + distance_atol,
            distance_order,
        )

    else:
        all_sites_extra = all_sites
        ids_extra = ids

        naive_edges_by_order = get_naive_edges(
            all_sites,
            distance_order * np.linalg.norm(basis_vectors, axis=1).max()
            + distance_atol,
            distance_order,
        )

    # Plot edges
    colors = get_n_colors(len(naive_edges_by_order))
    for k, naive_edges in enumerate(naive_edges_by_order):
        a = 1 - k * 0.2
        color = colors[k]
        for node1, node2 in naive_edges:
            # switch to real node indices
            pos1 = all_sites_extra[node1]
            pos2 = all_sites_extra[node2]

            # check if not within lattice
            pos1_id = lattice.positions[ids_extra[node1]]
            pos2_id = lattice.positions[ids_extra[node2]]

            if all(pos1_id == pos1) and all(pos2_id == pos2):
                linestyle = "-"
            else:
                linestyle = "--"

            if dim == 1:
                ax.plot(
                    [pos1[0], pos2[0]],
                    [0, 0],
                    linestyle=linestyle,
                    alpha=a,
                    color=color,
                )
            elif dim == 2:
                ax.plot(
                    [pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    linestyle=linestyle,
                    alpha=a,
                    color=color,
                )
            else:  # 3D
                ax.plot(
                    [pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    [pos1[2], pos2[2]],
                    linestyle=linestyle,
                    alpha=a,
                    color=color,
                )

    # Plot sites
    zorders = [
        artist.zorder for artist in ax.get_children() if hasattr(artist, "zorder")
    ]
    zorder = max(zorders) + 1

    if draw_neighbors:
        ax_scatter_autodim(
            ax,
            all_sites_extra,
            s=node_size,
            c=node_color,
            alpha=extra_sites_alpha,
            zorder=zorder,
        )
    ax_scatter_autodim(
        ax,
        all_sites,
        s=node_size,
        c=node_color,
        zorder=zorder + 1,
    )

    if draw_text:
        for pos, site in zip(all_sites_extra, ids_extra):
            ax_text_autodim(
                ax,
                pos,
                str(site),
                offset=node_text_offset,
                fontsize=8,
                horizontalalignment="center",
                verticalalignment="center",
                zorder=zorder + 2,
                color=node_text_color,
            )

    if draw_basis_vectors:
        # Plot basis vectors as arrows
        origin = np.zeros(dim)
        for i, basis in enumerate(basis_vectors):
            labl = f"$\\vec{{b}}_{{{i}}}$"
            if dim == 2:
                ax.quiver(
                    origin[0],
                    origin[1],
                    basis[0],
                    basis[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=f"C{i}",
                    width=0.005,
                    headwidth=10,
                    headlength=10,
                    zorder=-10,
                )
                midpoint = basis / 2
                ax.text(
                    midpoint[0],
                    midpoint[1],
                    labl,
                    fontsize=10,
                    ha="center",
                    va="center",
                    # bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )
            elif dim == 3:  # 3D
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    basis[0],
                    basis[1],
                    basis[2],
                    color=f"C{i}",
                    zorder=-10,
                )
                midpoint = basis / 2
                ax.text(midpoint[0], midpoint[1], midpoint[2], labl, fontsize=10)

    if draw_unit_cell:
        origin = np.zeros(dim)
        if dim == 2:
            # Shade the parallelogram formed by the basis vectors
            parallelogram = np.array(
                [
                    origin,
                    basis_vectors[0],
                    basis_vectors[0] + basis_vectors[1],
                    basis_vectors[1],
                ]
            )
            ax.add_patch(
                plt.Polygon(parallelogram, alpha=0.2, facecolor="gray", zorder=-20),
            )

        elif dim == 3:
            # Shade the parallelpiped formed by the basis vectors
            vertices = np.array(
                [
                    origin,
                    basis_vectors[0],
                    basis_vectors[1],
                    basis_vectors[2],
                    basis_vectors[0] + basis_vectors[1],
                    basis_vectors[0] + basis_vectors[2],
                    basis_vectors[1] + basis_vectors[2],
                    basis_vectors[0] + basis_vectors[1] + basis_vectors[2],
                ]
            )

            faces = [
                [vertices[0], vertices[1], vertices[4], vertices[2]],
                [vertices[0], vertices[1], vertices[5], vertices[3]],
                [vertices[0], vertices[2], vertices[6], vertices[3]],
                [vertices[7], vertices[4], vertices[1], vertices[5]],
                [vertices[7], vertices[4], vertices[2], vertices[6]],
                [vertices[7], vertices[5], vertices[3], vertices[6]],
            ]

            ax.add_collection3d(Poly3DCollection(faces, alpha=0.2, facecolor="gray"))

    # Set labels and title
    ax.set_xlabel("X")
    if dim >= 2:
        ax.set_ylabel("Y")
    if dim == 3:
        ax.set_zlabel("Z")
    ax.set_title(f"{dim}D Lattice (Distance Order: {distance_order})")

    if dim == 1:
        ax.set_ylim(-0.2, 0.2)
    elif dim == 2:
        ax.set_aspect("equal", "box")
    else:  # 3D
        # Equal aspect ratio for 3D plots
        ax.set_box_aspect((1, 1, 1))

    # Show the plot
    if show:
        plt.show()

    return ax
