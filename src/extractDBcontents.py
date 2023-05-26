from smartredis import Client
import numpy as np

# Initialize client
client = Client(cluster=False)
print("Initialized client")

# Define keys, loop through them extract data from DB and save it to file
training_data = [f"y.{i}" for i in range(24)]
keys = training_data
print(f"Extracting tensors with keys")
print(keys)

for key in keys:
    data = client.get_tensor(key)
    np.save(key,data)



