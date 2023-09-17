import pandas as pd
import json

with open('E:/JURIMIND/v0.0.1/ipc_reduced.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

df = pd.json_normalize(data)

df.to_csv('vectorized.csv', index=False)
