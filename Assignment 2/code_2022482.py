# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# Part (a) of question 1: Converting datatypes as necessary

# # converting IDs as string
# df_stops['stop_id'] = df_stops['stop_id'].astype(str)
# df_stops['zone_id'] = df_stops['zone_id'].astype(str)
# df_routes['route_id'] = df_routes['route_id'].astype(str)
# df_stop_times['stop_id'] = df_stop_times['stop_id'].astype(str)
# df_trips['route_id'] = df_trips['route_id'].astype(str)
# df_fare_rules['route_id'] = df_fare_rules['route_id'].astype(str)
# df_fare_rules['origin_id'] = df_fare_rules['origin_id'].astype(str)
# df_fare_rules['destination_id'] = df_fare_rules['destination_id'].astype(str)

# convert time columns as datetime 
df_stop_times['arrival_time'] = pd.to_datetime(df_stop_times['arrival_time'], format='%H:%M:%S', errors='coerce').dt.time
df_stop_times['departure_time'] = pd.to_datetime(df_stop_times['departure_time'], format='%H:%M:%S', errors='coerce').dt.time

# Part (b) is in the function below - creating the knowledge base

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    # print(len(df_trips))
    # print(df_trips.nunique())
    trip_to_route = dict(zip(df_trips['trip_id'], df_trips['route_id']))
    # print(trip_to_route)
    # print(len(trip_to_route))
    
    # Map route_id to a list of stops in order of their sequence
    route_to_stops = defaultdict(list)
    stop_sequence_dict = defaultdict(list)
    for trip_id, stop_id, stop_sequence in zip(df_stop_times['trip_id'], 
                                            df_stop_times['stop_id'], 
                                            df_stop_times['stop_sequence']):
        route_id = trip_to_route[str(trip_id)]
        stop_sequence_dict[route_id].append((stop_sequence, stop_id))

    # Sorting the stops by seq # & ensure unique stops
    for route_id, stop_sequence_list in stop_sequence_dict.items():
        sorted_stops = sorted(stop_sequence_list)
        unique_stops = [stop for _, stop in sorted_stops]
        route_to_stops[route_id] = list(dict.fromkeys(unique_stops))
    # print(route_to_stops)  
    
    # Count trips per stop
    stop_trip_count = defaultdict(int)
    # print('Count of unique values in stop_id:', df_stop_times['stop_id'].nunique())
    for stop in df_stop_times['stop_id']:
        stop_trip_count[stop] += 1
    # print(stop_trip_count)
    # print(len(stop_trip_count))

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id')
    # print(merged_fare_df)

    # Create fare rules for routes
    fare_rules = {}
    merged_fare_rules = merged_fare_df[['route_id', 'fare_id', 'origin_id', 'destination_id', 'price', 'currency_type', 'agency_id']]

    fare_rules = {
        route_id: {
            'fare_id': fare_id,
            'origin_id': origin_id,
            'destination_id': destination_id,
            'price': price,
            'currency': currency,
            'agency_id': agency_id
        }
        for route_id, fare_id, origin_id, destination_id, price, currency, agency_id in zip(
            merged_fare_rules['route_id'],
            merged_fare_rules['fare_id'],
            merged_fare_rules['origin_id'],
            merged_fare_rules['destination_id'],
            merged_fare_rules['price'],
            merged_fare_rules['currency_type'],
            merged_fare_rules['agency_id']
        )
    }
    # print(fare_rules)
    # print(len(fare_rules))

# create_kb()

# Part (c) are in the functions below - verifying the knowledge base

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """

    route_trip_count = defaultdict(int)
    for _, route_id in trip_to_route.items():
        route_trip_count[route_id] += 1

    busiest_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)
    return busiest_routes[:5]

# print(type(get_busiest_routes()))
# print(type(get_busiest_routes()[0]))
# print(get_busiest_routes())

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """

    most_frequent_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)
    return most_frequent_stops[:5]

# print(type(get_busiest_routes()))
# print(type(get_busiest_routes()[0]))
# print(get_most_frequent_stops())

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    
    stop_route_count = defaultdict(int)
    for route_id, stops in route_to_stops.items():
        for stop in stops:
            stop_route_count[stop] += 1

    busiest_stops = sorted(stop_route_count.items(), key=lambda x: x[1], reverse=True)
    return busiest_stops[:5]

# print(type(get_top_5_busiest_stops()))
# print(type(get_top_5_busiest_stops()[0]))
# print(get_top_5_busiest_stops())

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    
    stop_pairs = defaultdict(lambda: defaultdict(int))

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stop_1, stop_2 = stops[i], stops[i + 1]
            stop_pairs[(stop_1, stop_2)][route_id] += 1

    single_route_pairs = []
    for (stop_1, stop_2), routes in stop_pairs.items():
        if len(routes) == 1:  # Only one direct route 
            route_id = next(iter(routes)) 
            combined_trip_frequency = stop_trip_count[stop_1] + stop_trip_count[stop_2]
            single_route_pairs.append(((stop_1, stop_2), route_id, combined_trip_frequency))

    sorted_pairs = sorted(single_route_pairs, key=lambda x: x[2], reverse=True)

    return [(pair, route_id) for pair, route_id, _ in sorted_pairs[:5]]

# print(type(get_stops_with_one_direct_route()))
# print(type(get_stops_with_one_direct_route()[0]))
# print(get_stops_with_one_direct_route())

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    G = nx.Graph()

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id) 

    pos = nx.spring_layout(G, seed=42) 

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Stop ID: {node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Stop Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Interactive Stop-Route Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.show()

# visualize_stop_route_graph_interactive(route_to_stops)

# Part (a) of question 2: Brute-Force

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    
    direct_routes = [route_id for route_id, stops in route_to_stops.items() 
                     if start_stop in stops and end_stop in stops]
    
    return direct_routes

# print(direct_route_brute_force(100, 101))

# Part (b) of question 2: Datalog Reasoning

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    DirectRoute(X, Y, R) <= (RouteHasStop(R, X) & RouteHasStop(R, Y))
    OptimalRoute(X, Y, Z, R1, R2) <= (DirectRoute(X, Y, R1) & DirectRoute(Y, Z, R2) & (R1 != R2))

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    
    for route_id in route_to_stops:
        for stop in route_to_stops[route_id]:
            +RouteHasStop(route_id, stop)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    direct_routes = [route[0] for route in DirectRoute(start, end, R)]
    return sorted(direct_routes)

# initialize_datalog()
# print(query_direct_routes(100, 101))

# Part (a) of question 3: Forward Chaining

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    paths = OptimalRoute(start_stop_id, stop_id_to_include, end_stop_id, R1, R2)
    return [(path[0], stop_id_to_include, path[1]) for path in paths]


# Part (b) of question 3: Backward Chaining

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """

    paths = OptimalRoute(end_stop_id, stop_id_to_include, start_stop_id, R1, R2)
    return [(path[0], stop_id_to_include, path[1]) for path in paths]


# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    pass  # Implementation here

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
