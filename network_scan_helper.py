import time
import osmnx as ox # type: ignore
import geopandas as gpd # type: ignore
import warnings
warnings.filterwarnings('ignore')
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as mpatches # type: ignore
from matplotlib_scalebar.scalebar import ScaleBar # type: ignore
import contextily as cx # type: ignore
import numpy as np
import time
import networkx as nx # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore
import random
from joblib import Parallel, delayed # type: ignore
import os
import seaborn as sns # type: ignore
import ast
from shapely.geometry import Polygon, MultiPolygon, Point # type: ignore

def create_output_folders(city):
    folders = [f'output/{city}/png', f'output/{city}/tabular', f'output/{city}/geojson']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def get_AOI(gdf, city, proj_crs):
    create_output_folders(city=city)
    gdf_city = gdf[gdf.ADM3_EN == city]
    gdf_city_buffer = gdf_city.copy().to_crs(proj_crs)
    gdf_city_buffer['geometry'] = gdf_city_buffer['geometry'].buffer(1000)
    gdf_city_buffer = gdf_city_buffer.to_crs(4326)
    return gdf_city, gdf_city_buffer


def map_basemap(
    gdf_city,
    gdf_city_buffer,
    road_edges,
    road_nodes,
    crs=3121,
    basemap_provider=cx.providers.CartoDB.PositronNoLabels
):
    """
    Plots a basic styled map of AOI definition.

    Parameters:
        gdf_city (GeoDataFrame): GeoDataFrame of a particular city.
        gdf_city_buffer (GeoDataFrame): Boundary GeoDataFrame 1km buffer to be plotted.
        road_edges (GeoDataFrame): Network Edges to be plotted.
        road_nodes (GeoDataFrame): Network Nodes to be plotted.
        crs (int): Projected CRS in meters, default EPSG:3121.
        basemap_provider: Contextily basemap provider object.

    Returns:
        Matplotlib Axes object.
    """

    # Project to CRS
    gdf_city_proj = gdf_city.to_crs(crs)
    gdf_city_buffer_proj = gdf_city_buffer.to_crs(crs)
    road_edges_proj = road_edges.to_crs(crs)
    road_nodes_proj = road_nodes.to_crs(crs)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 20))

    # Add border
    gdf_city_proj.boundary.plot(ax=ax, edgecolor='red', linewidth=1.5, zorder=4)
    gdf_city_buffer_proj.boundary.plot(ax=ax, edgecolor='red', linewidth=0.8, linestyle='--', zorder=3)

    # Add edges & nodes
    road_edges_proj.plot(ax=ax, edgecolor='k', linewidth=0.3, zorder=1)
    road_nodes_proj.plot(ax=ax, edgecolor='k', markersize=0.15, color='white',zorder=2)

    # Add basemap
    cx.add_basemap(ax, source=basemap_provider, crs=crs, zoom=14)
    ax.set_axis_off()


    # Add scale bar
    scalebar = ScaleBar(
        dx=1, units='m', height_fraction=0.01, length_fraction=0.2,
        location='upper left', scale_loc='bottom', box_alpha=0, color='black'
    )
    ax.add_artist(scalebar)
    plt.close()

    return fig, ax

def extract_road_network(gdf_polygon, network_type='all', simplify=True, verbose=True):
    """
    Extract and process a road network from a given polygon using OSMnx.

    Parameters:
    ----------
    gdf_polygon : GeoDataFrame
        A GeoDataFrame containing the polygon geometry for which the road network is to be extracted.
    network_type : str, default 'all'
        The type of network to extract. Options include 'all', 'drive', 'walk', 'bike', etc.
    simplify : bool, default True
        Whether to simplify the graph topology.
    verbose : bool, default True
        If True, prints timing information for each step.

    Returns:
    -------
    road_graphs : networkx.MultiDiGraph
        The extracted road network graph.
    road_edges : GeoDataFrame
        GeoDataFrame of road edges with flattened 'highway' attributes.
    road_nodes : GeoDataFrame
        GeoDataFrame of road nodes.
    """

    def log(msg):
        if verbose:
            print(msg)

    total_start = time.time()

    # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)

        # 1. Extract the road network
        start = time.time()
        road_graphs = ox.graph_from_polygon(
            polygon=gdf_polygon.geometry.unary_union,
            network_type=network_type,
            simplify=simplify
        )
        log(f"Graph extraction took {time.time() - start:.2f} seconds")

        # 2. Convert to GeoDataFrames
        start = time.time()
        road_edges = ox.graph_to_gdfs(road_graphs, nodes=False, edges=True).reset_index()
        road_nodes = ox.graph_to_gdfs(road_graphs, nodes=True, edges=False).reset_index()
        log(f"Graph to GeoDataFrame conversion took {time.time() - start:.2f} seconds")

        # 3. Flatten highway lists
        start = time.time()
        road_edges['highway_str'] = road_edges['highway'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        road_edges['highway_primary'] = road_edges['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
        log(f"Flattening 'highway' values took {time.time() - start:.2f} seconds")

        # 4. Cleaning up categories
        start = time.time()

        # Mapping rules to simplify road hierarchy
        def simplify_highway(value):
            if isinstance(value, list):
                value = value[0]  # Take the first one if it's a list
            if value is None:
                return None
            if '_link' in value:
                value = value.replace('_link', '')
            if value in ['residential', 'living_street', 'unclassified']:
                return 'residential'
            elif value in ['pedestrian', 'footway', 'path', 'cycleway', 'steps', 'bridleway', 'corridor']:
                return 'pedestrian'
            return value

        # Apply simplification
        road_edges['highway_simplified'] = road_edges['highway_primary'].apply(simplify_highway)

        # Define simplified hierarchy
        simplified_hierarchy = [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'service', 'busway', 'pedestrian'
        ]

        # Set as ordered category
        road_edges['highway_cat'] = pd.Categorical(
            road_edges['highway_simplified'],
            categories=simplified_hierarchy,
            ordered=True
        )

        log(f"Simplifying road hierarchy took {time.time() - start:.2f} seconds")

        log(f"\nTotal time: {time.time() - total_start:.2f} seconds")

    return road_graphs, road_edges, road_nodes


def map_road_hierarchy(
    road_edges,
    gdf_border,
    crs=3121,
    colormap='plasma',
    basemap_provider=cx.providers.CartoDB.PositronNoLabels
):
    """
    Plots a styled map of road hierarchy with categorical color and variable linewidths.

    Parameters:
        road_edges (GeoDataFrame): GeoDataFrame containing the road segments with a 'highway_cat' column.
        gdf_border (GeoDataFrame): Boundary GeoDataFrame to be plotted.
        crs (int): Projected CRS in meters, default EPSG:3121.
        colormap (str): Name of a matplotlib colormap.
        basemap_provider: Contextily basemap provider object.

    Returns:
        Matplotlib Axes object.
    """

    road_hierarchy = [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'service', 'busway', 'pedestrian'
        ]

    # Ensure the column is a categorical type with the specified order
    road_edges['highway_cat'] = pd.Categorical(
        road_edges['highway_cat'], categories=road_hierarchy, ordered=True
    )

    # Color and linewidth dicts
    cmap = plt.get_cmap(colormap, len(road_hierarchy))
    color_dict = {cat: cmap(i) for i, cat in enumerate(road_hierarchy)}
    linewidths = np.linspace(2, 0.1, len(road_hierarchy))
    linewidth_dict = dict(zip(road_hierarchy, linewidths))

    # Project to CRS
    road_edges_proj = road_edges.to_crs(crs)
    gdf_border_proj = gdf_border.to_crs(crs)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 20))

    for cat in road_hierarchy:
        subset = road_edges_proj[road_edges_proj['highway_cat'] == cat]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color_dict[cat],
                linewidth=linewidth_dict[cat],
                label=cat
            )

    # Add border
    gdf_border_proj.boundary.plot(ax=ax, edgecolor='k', linewidth=0.8)

    # Add basemap
    cx.add_basemap(ax, source=basemap_provider, crs=crs, zoom=14)
    ax.set_axis_off()

    # Add custom legend
    legend_handles = [mpatches.Patch(color=color_dict[cat], label=cat) for cat in road_hierarchy]
    ax.legend(
        handles=legend_handles,
        title='Road Hierarchy',
        title_fontsize=12,
        fontsize=10,
        loc='lower right',
        bbox_to_anchor=(1.18, 0),
        frameon=False
    )

    # Add scale bar
    scalebar = ScaleBar(
        dx=1, units='m', height_fraction=0.01, length_fraction=0.2,
        location='upper left', scale_loc='bottom', box_alpha=0, color='black'
    )
    ax.add_artist(scalebar)
    plt.close()

    return fig, ax

def plot_total_length_by_cat(desc):
    fig,ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=desc, x='length_km', y=desc.index, ax=ax)
    plt.grid(visible=True, linewidth=0.4, linestyle='--')
    plt.xlabel('Total length (km)')
    plt.ylabel('Road Hierarchy')
    plt.title('Total Road Length by Category')
    plt.tight_layout()
    plt.close()
    return fig, ax

def plot_median_length_by_cat(desc):
    fig,ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=desc, x='50%', y=desc.index)
    plt.grid(visible=True, linewidth=0.4, linestyle='--')
    plt.xlabel('Median Segment Length (m)')
    plt.ylabel('Road Hierarchy')
    plt.title('Median Road Segment Length by Category')
    plt.tight_layout()
    plt.close()
    return fig, ax


def map_road_intersections(
    road_edges,
    road_nodes,
    gdf_border,
    crs=3121,
    colormap='plasma',
    basemap_provider=cx.providers.CartoDB.PositronNoLabels
):
    """
    Plots a styled map of road intersections with categorical color and variable marker size.

    Parameters:
        road_edges (GeoDataFrame): GeoDataFrame containing the road segments.
        road_nodes (GeoDataFrame): GeoDataFrame containing the intersection nodes with a 'street_count' column.
        gdf_border (GeoDataFrame): Boundary GeoDataFrame to be plotted.
        crs (int): Projected CRS in meters, default EPSG:3121.
        colormap (str): Name of a matplotlib colormap.
        basemap_provider: Contextily basemap provider object.

    Returns:
        Matplotlib Axes object.
    """
    # Project to CRS
    road_nodes_proj = road_nodes.to_crs(crs)
    road_edges_proj = road_edges.to_crs(crs)
    gdf_border_proj = gdf_border.to_crs(crs)    
    # making markersize
    road_nodes_proj['markersize'] = 2**road_nodes_proj['street_count']
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 20))
    # plot road edges
    road_edges_proj.plot(ax=ax,
                    linewidth= 0.2,
                    color = 'grey'
                    )
    # Add border
    gdf_border_proj.boundary.plot(ax=ax, edgecolor='k', linewidth=0.8)
    # plot road intersections
    road_nodes_proj.plot(ax=ax,
                         column='street_count', 
                         cmap=colormap,
                         categorical=True,
                         markersize='markersize', 
                         legend=True,
                         edgecolor='black', 
                         linewidth=0.5, 
                         alpha = 0.7
                         )
    # Add basemap
    cx.add_basemap(ax, source=basemap_provider, crs=crs, zoom=14)
    ax.set_axis_off()
    # Move legend to bottom right, outside map
    leg = ax.get_legend()
    if leg:
        leg.set_bbox_to_anchor((1.25, 0.2))  # (x, y) position outside bottom right
        leg.set_title("Street Count per Intersection")
        leg.set_frame_on(False)
        leg.set_alignment('left')
    # Add scale bar
    scalebar = ScaleBar(
        dx=1, units='m', height_fraction=0.01, length_fraction=0.2,
        location='upper left', scale_loc='bottom', box_alpha=0, color='black'
    )
    ax.add_artist(scalebar)
    plt.close()
    return fig, ax

def calc_basic_stats(graph):
    start = time.time()
    # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # calculate basic stats
        stats = ox.stats.basic_stats(graph)

        # Descriptions for each metric
        descriptions = {
            'n': 'Total number of nodes (e.g., intersections, dead-ends)',
            'm': 'Total number of edges (street segments)',
            'k_avg': 'Average node degree (connectivity of the graph)',
            'edge_length_total': 'Total length of all edges in meters',
            'edge_length_avg': 'Average length of each edge (segment)',
            'streets_per_node_avg': 'Average number of streets per node',
            'intersection_count': 'Total number of true intersections',
            'street_length_total': 'Total length of undirected street segments',
            'street_segment_count': 'Number of street segments (simplified, undirected)',
            'street_length_avg': 'Average length of each street segment',
            'circuity_avg': 'Average circuity (detour factor compared to straight line)',
            'self_loop_proportion': 'Proportion of edges that loop back to the same node',
            'streets_per_node_counts': 'Count of nodes with specific street counts',
            'streets_per_node_proportions': 'Proportion of nodes with specific street counts',
        }

        # Convert the stats and descriptions to a dataframe
        rows = []
        for key, value in stats.items():
            if isinstance(value, dict):
                rows.append([key, str(value), descriptions.get(key, '')])
            else:
                rows.append([key, round(value, 2), descriptions.get(key, '')])

        df_stats = pd.DataFrame(rows, columns=["Metric", "Value", "Description"])
        end = time.time()
        print(f"Calculating basic stats took {end - start:.2f} seconds")
        return df_stats
    
def compute_lanes_maxspeed(road_edges):
    road_edges['lanes_clean'] = None
    road_edges['maxspeed_clean'] = None
    for i in range(len(road_edges)):
        val = road_edges['lanes'].iloc[i]
        if isinstance(val, list):
            # Convert string values to float or int
            try:
                road_edges.at[i, 'lanes_clean'] = int(min([float(x) for x in val]))
            except:
                continue
        elif pd.isna(val):
            continue
        else:
            try:
                road_edges.at[i, 'lanes_clean'] = int(float(val))
            except:
                continue
    for i in range(len(road_edges)):
        val = road_edges['maxspeed'].iloc[i]
        if isinstance(val, list):
            # Convert string values to float or int
            try:
                road_edges.at[i, 'maxspeed_clean'] = int(min([float(x) for x in val]))
            except:
                continue
        elif pd.isna(val):
            continue
        else:
            try:
                road_edges.at[i, 'maxspeed_clean'] = int(float(val))
            except:
                continue
    road_edges['lanes_clean']= road_edges['lanes_clean'].astype('Int64')
    road_edges['maxspeed_clean']= road_edges['maxspeed_clean'].astype('Int64')
    lanes_speed_df = road_edges.groupby(by='highway_cat')[['lanes_clean', 'maxspeed_clean']].median().reset_index()
    lanes_speed_df.columns = ['highway_cat', 'median_lanes', 'median_maxspeed']
    lanes_sort = lanes_speed_df.dropna(subset=['median_lanes'])
    speed_sort = lanes_speed_df.dropna(subset=['median_maxspeed'])
    return lanes_speed_df, lanes_sort, speed_sort

def plot_median_lanes(lanes_sort):
    # Plot figures
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=lanes_sort, y='highway_cat', x='median_lanes')
    plt.title('Median Lanes by Road Hierarchy')
    plt.xlabel('Median Lanes')
    plt.xticks(np.linspace(0,4,num=5))
    plt.ylabel('Road Category')
    plt.grid(which='both', axis='x', linestyle= '--')
    plt.close()
    return fig, ax

def plot_median_speed(speed_sort):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=speed_sort, y='highway_cat', x='median_maxspeed')
    plt.title('Median Maxspeed by Road Hierarchy')
    plt.xlabel('Median Maxspeed (km/h)')
    plt.ylabel('Road Category')
    plt.grid(which='both', axis='x', linestyle= '--')
    plt.close()
    return fig, ax

def calculate_intersection(df_stats):
    # Get the string from the dataframe and convert it to a dict
    streets_per_node_counts_str = df_stats[df_stats.Metric == 'streets_per_node_counts'].Value.values[0]
    streets_per_node_counts = ast.literal_eval(streets_per_node_counts_str)

    # Description mapping
    descriptions = {
        0: "Isolated node (possible data error)",
        1: "Dead ends",
        2: "Mid-block geometry point, i.e., bend/curve",
        3: "3-way intersection (T-junction)",
        4: "4-way intersection (crossroad)",
        5: "5-way intersection",
        6: "6-way intersection"
    }

    # Create dataframe
    street_node_df = pd.DataFrame([
        {"streets_per_node": k, "description": descriptions.get(k, "Other"), "count": v}
        for k, v in streets_per_node_counts.items()
    ])

    # Sort and display
    street_node_df = street_node_df.sort_values(by="streets_per_node").reset_index(drop=True)
    return street_node_df

def compute_closeness_node(G, node):
    return node, nx.closeness_centrality(G, u=node, distance='length')

def compute_graph_centralities(
    graph,
    degree_centrality=True,
    node_closeness_centrality=True,
    node_betweenness_centrality=True,
    edge_betweenness_centrality=True,
    k=None,
    seed=None
):
    """
    Compute selected centrality measures for a given OSMnx graph.

    Parameters:
        graph (networkx.MultiDiGraph): The road network graph.
        degree_centrality (bool): If True, computes node degree centrality.
        node_closeness_centrality (bool): If True, computes node closeness centrality.
        node_betweenness_centrality (bool): If True, computes node betweenness centrality.
        edge_betweenness_centrality (bool): If True, computes edge betweenness centrality.
        k (int or None): Number of samples for approximation in betweenness centrality. If None, full calculation is done.
        seed (int or None): Random seed for reproducibility when using `k`.

    Returns:
        nodes_gdf (GeoDataFrame): Nodes with selected centrality measures.
        edges_gdf (GeoDataFrame): Edges with edge betweenness if selected.
    """
    start = time.time()
    # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        
        # Ensure the graph is undirected
        G = ox.convert.to_undirected(graph)

        # --- Node Centrality Computation ---
        deg_cent = {}
        close_cent = {}
        btwn_cent = {}

        if degree_centrality:
            deg_cent = nx.degree_centrality(G)
            end = time.time()
            print(f"✔ Node Degree Centrality computed: {end - start:.2f} seconds")

        if node_closeness_centrality:
                if k is not None:
                    rng = random.Random(seed)
                    sample_nodes = rng.sample(list(G.nodes), k=min(k, len(G.nodes)))
                    results = Parallel(n_jobs=-1)(
                        delayed(compute_closeness_node)(G, n) for n in sample_nodes
                    )
                    close_cent = dict(results)
                    end = time.time()
                    print(f"✔ Node Closeness Centrality computed (sampled: {len(close_cent)} nodes): {end - start:.2f} seconds")
                else:
                    close_cent = nx.closeness_centrality(G, distance='length')
                    end = time.time()
                    print(f"✔ Node Closeness Centrality computed (full graph): {end - start:.2f} seconds")
        if node_betweenness_centrality:
            btwn_cent = nx.betweenness_centrality(
                G, weight='length', normalized=True, k=k, seed=seed
            )
            end = time.time()
            print(f"✔ Node Betweenness Centrality computed: {end - start:.2f} seconds")

        # Convert graph to GeoDataFrame (nodes only)
        nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)

        # Add centralities to nodes GeoDataFrame
        if deg_cent:
            nodes_gdf['degree_centrality'] = nodes_gdf.index.map(deg_cent)
        if close_cent:
            nodes_gdf['closeness_centrality'] = nodes_gdf.index.map(close_cent)
        if btwn_cent:
            nodes_gdf['betweenness_centrality'] = nodes_gdf.index.map(btwn_cent)

        # --- Edge Centrality Computation ---
        edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)

        if edge_betweenness_centrality:
            edge_btwn_cent = nx.edge_betweenness_centrality(
                G, weight='length', normalized=True, k=k, seed=seed
            )
            # Convert to Series and align by edge keys (u, v, key)
            edge_btwn_series = pd.Series(edge_btwn_cent)
            edges_gdf['edge_betweenness'] = edge_btwn_series
            end = time.time()
            print(f"✔ Edge Betweenness Centrality computed: {end - start:.2f} seconds")

        end = time.time()
        print(f"✅ Total computation time: {end - start:.2f} seconds")

    return nodes_gdf, edges_gdf


def map_nodes_centrality(
    road_edges,
    road_nodes,
    gdf_border,
    column,
    crs=3121,
    colormap='plasma',
    basemap_provider=cx.providers.CartoDB.PositronNoLabels
):
    """
    Plots a styled map of road intersections with categorical color and variable marker size.

    Parameters:
        road_edges (GeoDataFrame): GeoDataFrame containing the road segments.
        road_nodes (GeoDataFrame): GeoDataFrame containing the intersection nodes with centrality values column to be visualized.
        gdf_border (GeoDataFrame): Boundary GeoDataFrame to be plotted.
        crs (int): Projected CRS in meters, default EPSG:3121.
        colormap (str): Name of a matplotlib colormap.
        basemap_provider: Contextily basemap provider object.

    Returns:
        Matplotlib Axes object.
    """
     # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Project to CRS
        road_nodes_proj = road_nodes.to_crs(crs)
        road_edges_proj = road_edges.to_crs(crs)
        gdf_border_proj = gdf_border.to_crs(crs)    
        
        # Scale for value readability (-1 to 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized = scaler.fit_transform(road_nodes_proj[[column]]).flatten()
        # Rescale from [0, 1] to [-1, 1]
        road_nodes_proj['norm_value'] = normalized * 2 - 1
        # making markersize
        val_column = road_nodes_proj[column].values.reshape(-1, 1)
        # Scale for markersize between 1 and 10
        markerscaler = MinMaxScaler(feature_range=(1, 25))
        scaled_vals = markerscaler.fit_transform(val_column).flatten()
        road_nodes_proj['markersize'] = scaled_vals
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        # plot road edges
        road_edges_proj.plot(ax=ax,
                        linewidth= 0.2,
                        color = 'grey'
                        )
        # Add border
        gdf_border_proj.boundary.plot(ax=ax, edgecolor='k', linewidth=0.8)
        # plot road intersections
        road_nodes_proj.plot(ax=ax,
                            column='norm_value', 
                            cmap=colormap,
                            categorical=False,
                            markersize='markersize', 
                            legend=True,
                            cax=cax,
                            edgecolor='black', 
                            linewidth=0.5, 
                            alpha = 0.7,
                            )
        # Add basemap
        cx.add_basemap(ax, source=basemap_provider, crs=crs, zoom=14)
        ax.set_axis_off()
        
        # Add scale bar
        scalebar = ScaleBar(
            dx=1, units='m', height_fraction=0.01, length_fraction=0.2,
            location='upper left', scale_loc='bottom', box_alpha=0, color='black'
        )
        ax.add_artist(scalebar)
        plt.close()
        return fig, ax
    


def map_edges_centrality(
    road_edges,
    road_nodes,
    gdf_border,
    crs=3121,
    colormap='plasma',
    basemap_provider=cx.providers.CartoDB.PositronNoLabels
):
    """
    Plots a styled map of road intersections with categorical color and variable marker size.

    Parameters:
        road_edges (GeoDataFrame): GeoDataFrame containing the road segments with betweenness_centrality column.
        road_nodes (GeoDataFrame): GeoDataFrame containing the intersection nodes.
        gdf_border (GeoDataFrame): Boundary GeoDataFrame to be plotted.
        crs (int): Projected CRS in meters, default EPSG:3121.
        colormap (str): Name of a matplotlib colormap.
        basemap_provider: Contextily basemap provider object.

    Returns:
        Matplotlib Axes object.
    """
     # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Project to CRS
        road_nodes_proj = road_nodes.to_crs(crs)
        road_edges_proj = road_edges.to_crs(crs)
        gdf_border_proj = gdf_border.to_crs(crs)    
        
        # Scale for value readability (-1 to 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized = scaler.fit_transform(road_edges_proj[['edge_betweenness']]).flatten()
        # Rescale from [0, 1] to [-1, 1]
        road_edges_proj['norm_value'] = normalized * 2 - 1
        # making markersize
        val_column = road_edges_proj['edge_betweenness'].values.reshape(-1, 1)
        # Scale for markersize between 1 and 10
        markerscaler = MinMaxScaler(feature_range=(1, 25))
        scaled_vals = markerscaler.fit_transform(road_edges_proj[['edge_betweenness']]).flatten()
        road_edges_proj['markersize'] = scaled_vals
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        # Add border
        gdf_border_proj.boundary.plot(ax=ax, edgecolor='k', linewidth=0.8)
        # plot road intersections
        road_edges_proj.plot(ax=ax,
                            column='norm_value', 
                            cmap=colormap,
                            categorical=False,
                            legend=True,
                            cax=cax,
                            linewidth=road_edges_proj['edge_betweenness']*10**3, 
                            alpha = 0.7,
                            )
        # Add basemap
        cx.add_basemap(ax, source=basemap_provider, crs=crs, zoom=14)
        ax.set_axis_off()
        
        # Add scale bar
        scalebar = ScaleBar(
            dx=1, units='m', height_fraction=0.01, length_fraction=0.2,
            location='upper left', scale_loc='bottom', box_alpha=0, color='black'
        )
        ax.add_artist(scalebar)
        plt.close()
        return fig, ax

def interpolate_polygon_edges(gdf, spacing_meters=25, crs=3121):
    # Reproject to a CRS in meters for accurate distance measurements
    gdf = gdf.to_crs(crs)

    points = []

    for _, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or isinstance(geom, Point):
            continue  # skip points

        # Get all exterior boundaries for Polygon or MultiPolygon
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        else:
            continue  # skip unknown geometry types

        for poly in polygons:
            boundary = poly.exterior
            length = boundary.length
            num_points = int(length // spacing_meters)

            for i in range(num_points + 1):
                point = boundary.interpolate(i * spacing_meters)
                points.append({
                    'geometry': point,
                    'source_name': row.get('name', None),
                    'source_index': row.name
                })

    # Return as GeoDataFrame (reprojected back to WGS84)
    points_gdf = gpd.GeoDataFrame(points, geometry='geometry', crs=gdf.crs)
    return points_gdf.to_crs(epsg=4326)

def extract_POI(tags, gdf_city_buffer, interpolation):
    """
    Extract POIs from OSM.

    Parameters:
    -----------
    tags : dict
        dictionary of OSM values to be extracted.
    gdf_city_buffer : GeoDataFrame
        Buffered GeoDataFrame as a bounding box in which POIs are extracted from.
    interpolation : Boolean
        If True, point interpolation for every 200m is generated as POI. Else, only centroid is generated. This is useful so that large POIs such as green open space are not reduced to a single point. 

    Returns:
    --------
    POI_gdf : GeoDataFrame
        Nodes of POIs for accessibility analysis.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        POI_gdf = ox.geometries_from_polygon(polygon=gdf_city_buffer.geometry.values[0], tags=tags)
        # Ensure only relevant columns are kept
        # Ensure all expected columns are included, even if missing ---
        desired_cols = ['name'] + list(tags.keys()) + ['geometry']
        POI_gdf = POI_gdf.reindex(columns=desired_cols)
        # Convert all geometries to centroids
        if interpolation == False:
            POI_gdf['geometry'] = POI_gdf.centroid
            POI_gdf['center'] = POI_gdf.geometry.apply(lambda point: [point.x, point.y])
            POI_gdf = POI_gdf.reset_index()
        else:
            POI_gdf = interpolate_polygon_edges(POI_gdf, spacing_meters=200, crs=3121)
            POI_gdf = POI_gdf.reset_index()
    
        return POI_gdf
    

    

def make_isochrone(graph, origin_gdf, distance_list, buffer_size=0.0006):
    """
    Generate isochrone buffers from origin points on a network graph.

    Parameters:
    -----------
    graph : networkx.Graph or MultiGraph
        The street network graph.
    origin_gdf : GeoDataFrame
        GeoDataFrame of origin points (e.g., schools) with Point geometries.
    distance_list : list
        List of distance cutoffs (e.g., [400, 800, 1600]).
    buffer_size : float
        Buffer size in coordinate units (default: 0.0006 for EPSG:4326).

    Returns:
    --------
    nodes_access : GeoDataFrame
        Nodes with distance to nearest origin.
    edges_access : GeoDataFrame
        Edges annotated with nearest-origin distance (via joined nodes).
    isochrone_buffer : GeoDataFrame
        Buffer polygons for each distance in distance_list.
    """
    # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
    
        # Convert to undirected for shortest paths
        G = ox.convert.to_undirected(graph)

        # Get nodes and edges
        nodes_access, edges_access = ox.graph_to_gdfs(G)

        # Snap origin points to nearest graph nodes
        origin_gdf = origin_gdf.copy()
        origin_gdf['nearest_node'] = origin_gdf.geometry.apply(
            lambda x: ox.distance.nearest_nodes(G, x.x, x.y)
        )
        target_nodes = origin_gdf['nearest_node'].unique()

        # Compute shortest path lengths from each origin node
        lengths = {}
        for target in target_nodes:
            sp = nx.single_source_dijkstra_path_length(G, target, weight='length')
            for node, dist in sp.items():
                if node not in lengths or dist < lengths[node]:
                    lengths[node] = dist

        # Map distance to each node
        nodes_access['distance_to_nearest_origin'] = nodes_access.index.map(lengths)

        # Spatial join: assign nearest-origin distance to edges by intersecting nodes
        edges_access = edges_access.sjoin(
            nodes_access[['distance_to_nearest_origin', 'geometry']],
            how='left',
            predicate='intersects'
        )

        # Create isochrone buffers
        buffer_list = []
        for d in distance_list:
            filtered = edges_access[edges_access['distance_to_nearest_origin'] <= d]
            if not filtered.empty:
                buffer_geom = filtered.buffer(buffer_size).unary_union
                buffer_list.append({'geometry': buffer_geom, 'distance': d})

        isochrone_buffer = gpd.GeoDataFrame(buffer_list, crs=edges_access.crs)

        return nodes_access, edges_access, isochrone_buffer


def calc_isochrone_coverage(isochrones, gdf_city, crs = 3121):
    isochrones_3121 = isochrones.to_crs(crs).clip(gdf_city.to_crs(crs));
    isochrones_3121['area_sqkm'] = isochrones_3121.area/10**6;
    total_area = gdf_city.to_crs(crs).area.values[0]/10**6;
    isochrones_3121['percent_covered'] = isochrones_3121['area_sqkm']/total_area;
    isochrones_3121 = isochrones_3121.sort_values(by='distance', ascending=True).drop(columns='geometry');
    return isochrones_3121

def map_accessibility(
    road_edges,
    isochrone,
    destination,
    gdf_border,
    crs=3121,
    colormap='plasma',
    basemap_provider=cx.providers.CartoDB.PositronNoLabels
):
    """
    Plots a styled map of road intersections with categorical color and variable marker size.

    Parameters:
        road_edges (GeoDataFrame): GeoDataFrame containing the road segments.
        isochrone (GeoDataFrame): GeoDataFrame of isochrone buffers.
        destination (GeoDataFrame): GeoDataFrame containing the Point of Interest as origin point.
        gdf_border (GeoDataFrame): Boundary GeoDataFrame to be plotted.
        crs (int): Projected CRS in meters, default EPSG:3121.
        colormap (str): Name of a matplotlib colormap.
        basemap_provider: Contextily basemap provider object.

    Returns:
        Matplotlib Axes object.
    """
     # Suppress DeprecationWarnings within this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # Project to CRS
        destination_proj = destination.to_crs(crs)
        road_edges_proj = road_edges.to_crs(crs).reset_index()
        isochrone_proj = isochrone.to_crs(crs)
        gdf_border_proj = gdf_border.to_crs(crs)    
        

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        
        # plot road edges <=3200
        road_edges_proj[road_edges_proj.distance_to_nearest_origin <=3200].plot(ax=ax,
                            column='distance_to_nearest_origin', 
                            cmap=colormap,
                            categorical=False,
                            linewidth=1, 
                            legend=True,
                            cax=cax,
                            alpha = 1,
                            zorder=2,
                            )
        # plot road edges >3200
        road_edges_proj[road_edges_proj.distance_to_nearest_origin >3200].plot(ax=ax,
                            color='grey',
                            categorical=False,
                            linewidth=1, 
                            legend=False,
                            cax=cax,
                            alpha = 1,
                            zorder=3,
                            )
        # plot isochrones
        isochrone_proj.sort_values(by='distance', ascending=False).plot(ax=ax,
                            column='distance',
                            cmap=colormap,
                            markersize=1, 
                            legend=False,
                            alpha = 0.2,
                            zorder=1
                            )
        
        # plot destination points
        destination_proj.plot(ax=ax,
                            color='black',
                            markersize=4, 
                            legend=True,
                            alpha = 1,
                            zorder=4
                            )
        
        # Add border
        gdf_border_proj.boundary.plot(ax=ax, edgecolor='k', linewidth=0.8)
        # Add basemap
        cx.add_basemap(ax, source=basemap_provider, crs=crs, zoom=14)
        ax.set_axis_off()
        
        # Add scale bar
        scalebar = ScaleBar(
            dx=1, units='m', height_fraction=0.01, length_fraction=0.2,
            location='upper left', scale_loc='bottom', box_alpha=0, color='black'
        )
        ax.add_artist(scalebar)
        plt.close()
        return fig, ax

def sanitize_gdf_for_export(gdf):
    # Work on a copy so the original remains unaltered
    gdf_clean = gdf.copy()

    # Convert categorical columns to string
    for col in gdf_clean.select_dtypes(include='category').columns:
        gdf_clean[col] = gdf_clean[col].astype(str)

    # Convert list-type columns to string
    for col in gdf_clean.columns:
        if gdf_clean[col].apply(lambda x: isinstance(x, list)).any():
            gdf_clean[col] = gdf_clean[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

    # Optional: Drop columns that are still problematic (e.g., dicts)
    for col in gdf_clean.columns:
        if gdf_clean[col].apply(lambda x: isinstance(x, dict)).any():
            print(f"Warning: dropping column '{col}' because it contains dicts")
            gdf_clean = gdf_clean.drop(columns=[col])

    return gdf_clean

def analyze_accessibility(category, tags, road_graph, gdf_city_buffer, gdf_city, city, interpolation):
    print(f'accessibility to {category} facilities')

    # Extract POIs
    poi_gdf = extract_POI(tags=tags, gdf_city_buffer=gdf_city_buffer, interpolation=interpolation)

    # Generate isochrones and accessibility graph
    distances = [400, 800, 1600, 2400, 3200]
    nodes, edges, isochrones = make_isochrone(road_graph, poi_gdf, distances)
    # exporting geojson
    nodes_export = sanitize_gdf_for_export(nodes)
    nodes_export.to_file(f'output/{city}/geojson/{category}_nodes_accessibility.geojson')
    edges_export = sanitize_gdf_for_export(edges)
    edges_export.to_file(f'output/{city}/geojson/{category}_edges_accessibility.geojson')
    isochrones_export = sanitize_gdf_for_export(isochrones)
    isochrones_export.to_file(f'output/{city}/geojson/{category}_isochrones_accessibility.geojson')
    
    # Export distribution of distances
    nodes[['distance_to_nearest_origin']].describe().to_csv(f'output/{city}/tabular/distribution_of_access_to_{category}.csv')

    # Calculate area coverage
    coverage_df = calc_isochrone_coverage(isochrones=isochrones, gdf_city=gdf_city)
    coverage_df.to_csv(f'output/{city}/tabular/isochrone_coverage_{category}.csv')

    # Generate map
    fig, ax = map_accessibility(
        road_edges=edges.clip(gdf_city),
        destination=poi_gdf.clip(gdf_city),
        isochrone=isochrones.clip(gdf_city),
        gdf_border=gdf_city,
        crs=3121,
        colormap='RdYlGn_r'
    )
    fig.savefig(f'output/{city}/png/map_accessibility_{category}.png', dpi=300, bbox_inches='tight')


