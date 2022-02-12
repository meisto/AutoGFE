# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 09:34:26 PM CET
# Description: -
# ======================================================================
import numpy as np
import pandas as pd
 


def get_column_representation(
    series: pd.Series,
    num_samples: int,
    replacing: bool = True
):
    """
        TODO: documentation 
    """

    return series\
        .sample(num_samples, replace=replacing)\
        .sort_index(ascending=True)
