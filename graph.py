import pandas as pd
import networkx as nx
import pickle
import networkx as nx
import pandas as pd

df = pd.read_csv("Dataset/Base_with_identifiers.csv")

G = nx.Graph()

# Add nodes
for idx, row in df.iterrows():
    G.add_node(row["user_id"], fraud=row["fraud_bool"])

# Add edges based on shared identifiers
identifier_cols = ["email", "phone_number", "device_id", "ip_address"]

for col in identifier_cols:
    shared = df.groupby(col)["user_id"].apply(list)
    for user_list in shared:
        for i in range(len(user_list)):
            for j in range(i + 1, len(user_list)):
                G.add_edge(user_list[i], user_list[j], link=col)

with open("models/user_graph.gpickle", "wb") as f:
    pickle.dump(G, f)
