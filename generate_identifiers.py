import pandas as pd
import random
import numpy as np

df = pd.read_csv("Dataset/Base.csv")

num_rows = len(df)
random.seed(42)

emails = [f"user{random.randint(0, 3000)}@example.com" for _ in range(num_rows)]
phones = [f"+91{random.randint(7000000000, 9999999999)}" for _ in range(3000)]
devices = [f"device_{random.randint(0, 1500)}" for _ in range(3000)]
ips = [f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}" for _ in range(1000)]

df["email"] = random.choices(emails, k=num_rows)
df["phone_number"] = random.choices(phones, k=num_rows)
df["device_id"] = random.choices(devices, k=num_rows)
df["ip_address"] = random.choices(ips, k=num_rows)

df.to_csv("Dataset/Base_with_identifiers.csv", index=False)

print("Base_with_identifiers.csv created with synthetic identifiers.")
