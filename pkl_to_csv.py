import pickle as pkl
import pandas as pd

with open("data/option_data.pkl", "rb") as f:
    file = pkl.load(f)
    
df = pd.DataFrame(file)
df.to_csv("data/option_data.csv")
