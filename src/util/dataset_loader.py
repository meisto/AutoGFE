from typing import List

import os

import pandas as pd
import numpy as np

from src.env.feature import Dataset
from src.util.importer import generate_dataset

def load_datasets(path) -> List[Dataset]:
    """
        Load all datasets from path. A completely specified dataset consists
        of a file dataset.csv containing the datapoints and dataset.labels.csv
        containing the labels.
    """

    files = next(os.walk(path))[2]
    files = [x for x in files if x.endswith(".csv")]

    label_files = [x for x in files if x.endswith(".labels.csv")]
    ds_files = [f"{x[:-11]}.csv" for x in label_files]

    datasets = []
    for ds_file in ds_files:
        dataset_name = ds_file[:-4]

        label_file = f"{dataset_name}.labels.csv"
        assert label_file in label_files, f"No file '{label_file}' with labels."

        # Prepend path
        label_file = os.path.join(path, label_file)
        ds_file = os.path.join(path, ds_file)

        # Read labels and dataframe
        df = pd.read_csv(ds_file, index_col=False)
        labels: pd.DataFrame = pd.read_csv(label_file, index_col=False)

        assert labels.shape[1] == 1, f"Illegal labels in file '{label_file}'."
        labels = labels.iloc[:,0]

        # Generate dataset and values
        dataset, values = generate_dataset(df, labels, dataset_name)
        datasets.append((dataset,values))
        
    # Assert that there are no invalid values in the datasets
    for ds, values in datasets:

        for key in values.keys():
            x = values[key]
            assert not np.isnan(x).any(),\
                    f"[ERROR] NaN in dataset '{ds.name}', key '{key}'"
            assert not np.isinf(x).any(),\
                    f"[ERROR] INF in dataset '{ds.name}', key '{key}'"


    return datasets

