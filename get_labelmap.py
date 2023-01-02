"""
get python dictionary:
- key: validation image name
- value: class_id e.g. n04597913
TODO add class_name
"""
import os
import pandas as pd

def get_labelmap(csv_path):
    df = pd.read_csv(csv_path)
    df['classes'] = df['PredictionString'].apply(lambda x: x.split()[0])  # e.g. 'n04597913'
    labelmap = {}
    for x, y in zip(df['ImageId'].tolist(), df['classes'].tolist()):
        labelmap[x] = y
    return labelmap

if __name__ == "__main__":
    base_dir = "/home/pymi/dataset/ILSVRC_eval"
    labelmap = get_labelmap(os.path.join(base_dir, "LOC_val_solution.csv"))
    print(labelmap)
