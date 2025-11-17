import numpy as np
from reasoning.generate_examples import load_locations
import pandas as pd
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from PIL import Image

coordinates=[11,15,44.5,48]

def plot_city_connections_on_image(image_path, city_locs, adjacency, save_path=None, linewidth=2):
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open(image_path)
    W, H = img.size

    # prepare axis with matching coordinates
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)

    # display image exactly at pixel coords
    ax.imshow(img, extent=[0, W, H, 0])     # IMPORTANT

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                       # IMPORTANT
    ax.set_aspect('equal')                  # IMPORTANT
    ax.set_axis_off()

    # plot edges
    for city, neighbors in adjacency.items():
        x1, y1 = city_locs[city]
        for nb in neighbors:
            x2, y2 = city_locs[nb]
            ax.plot([x1, x2], [y1, y2], '-', color='white', linewidth=linewidth)

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def build_delaunay_adjacency_filtered(city_locs, length_factor=1.8):
    """
    Build a filtered Delaunay adjacency graph.

    Long edges (outliers) are removed using a threshold:
        L(edge) <= length_factor * median_edge_length

    Parameters:
        city_locs : dict {city: (x,y)} or DataFrame with index=city, columns=[x,y]
        length_factor : multiplier for filtering (typical 1.5–2.5)

    Returns:
        adjacency : dict {city_name: set(neighbor_cities)}
    """

    # -------------------------------------------------------
    # 1. Convert input to consistent arrays
    # -------------------------------------------------------
    if isinstance(city_locs, pd.DataFrame):
        coords = city_locs.values
        names = list(city_locs.index)
    else:
        names = list(city_locs.keys())
        coords = np.array([city_locs[name] for name in names], dtype=float)

    coords = np.asarray(coords, dtype=float)

    # -------------------------------------------------------
    # 2. Delaunay triangulation
    # -------------------------------------------------------
    tri = Delaunay(coords)

    # Collect all Delaunay edges (undirected)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = simplex[i]
            b = simplex[(i+1) % 3]
            edge = tuple(sorted((a, b)))
            edges.add(edge)

    # -------------------------------------------------------
    # 3. Compute lengths for filtering
    # -------------------------------------------------------
    lengths = np.array([
        np.linalg.norm(coords[a] - coords[b])
        for a, b in edges
    ])

    median_len = np.median(lengths)
    max_len = median_len * length_factor

    # -------------------------------------------------------
    # 4. Build adjacency with filtering
    # -------------------------------------------------------
    adjacency = {name: set() for name in names}

    for (a, b), L in zip(edges, lengths):
        if L <= max_len:  # keep only "reasonable" edges
            city_a = names[a]
            city_b = names[b]
            adjacency[city_a].add(city_b)
            adjacency[city_b].add(city_a)

    return adjacency


def to_test(base_path, coordinates):
    
    path="./reasoning/screen.png"
    # 2. Get the neighborhoods
    locations_name_px_pos = load_locations(coordinates,base_path)
    adj = build_delaunay_adjacency_filtered(locations_name_px_pos)  # a = your DataFrame/dict

    for city, neighbors in adj.items():
        print(f"{city} → {sorted(neighbors)}")

    plot_city_connections_on_image(
        image_path=path,
        city_locs=locations_name_px_pos,
        adjacency=adj,
        save_path="triangulation_overlay.png"
    )