
import networkx as nx
import pandas as pd
import pickle

def load_graph(path="models/user_graph.gpickle"):
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def extract_graph_features_for_row(G, row):
    node = f"user_{row['email']}" 

    if node not in G:
        return {
            "fraud_neighbors": 0,
            "fraud_ratio_neighbors": 0.0,
            "component_size": 1,
        }

    neighbors = list(G.neighbors(node))
    fraud_neighbors = [n for n in neighbors if G.nodes[n].get("fraud", 0) == 1]
    component_size = len(nx.node_connected_component(G, node))

    return {
        "fraud_neighbors": len(fraud_neighbors),
        "fraud_ratio_neighbors": len(fraud_neighbors) / len(neighbors) if neighbors else 0.0,
        "component_size": component_size,
    }

def extract_single_user_graph_features(G, user_row):
    identifiers = [("email", user_row["email"]),
                   ("phone_number", user_row["phone_number"]),
                   ("device_id", user_row["device_id"]),
                   ("ip_address", user_row["ip_address"])]

    user_node = None
    for prefix, value in identifiers:
        candidate = f"user_{value}"
        if candidate in G:
            user_node = candidate
            break

    if user_node is None:
        return {
            "graph_score": 0.0
        }

    neighbors = list(G.neighbors(user_node))
    fraud_neighbors = [n for n in neighbors if G.nodes[n].get("fraud", 0) == 1]
    score = len(fraud_neighbors) / len(neighbors) if neighbors else 0.0

    return {
        "graph_score": score
    }
