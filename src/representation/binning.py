# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Thu 09 Sep 2021 12:08:34 PM CEST
# Description: A naive approach to table flattening
# ======================================================================
import numpy as np
import pandas as pd
 


def get_column_representation(
    series: pd.Series,
    num_bins: int,
    additional_information: bool = False
):
    """
        TODO: documentation 
    """
    # Allow only float or int as dtype
    assert series.dtype == float or series.dtype == int

    min_element: float = float(series.min()) - 10e-3
    max_element: float = float(series.max()) + 10e-3


    if min_element - max_element < 10e-2:
        print("here")
        bin_size = 1.0 / num_bins
        labels = [min_element + i * bin_size for i in range(num_bins)]
    
    else:
        print("not here")
        bin_size= (max_element - min_element) / num_bins
        labels = [min_element + i * bin_size for i in range(num_bins)]

    try:

        bins = pd.cut(series, bins=num_bins, labels=labels, ordered=True)
        bins = bins.value_counts()
        bins: pd.Series = bins.sort_index(ascending=True)

    except:
        print("Error")
        print("Num bins ", num_bins)
        print("Min element ", min_element)
        print("Max element ", max_element)
        print("BIn size ", bin_size)
        print(series.value_counts())
        print(series)
        print()
        raise Exception("Here")

    if additional_information == True:
        representation = np.concatenate([
            np.array([min_element, max_element, bin_size]),
            bins.values
        ])
    else:
        representation = bins.values

    # Pad representation with zero when too short (e.g. all input values same).
    if len(representation) < num_bins:
        dif = num_bins - len(representation)
        representation = np.concatenate([representation, [0] * dif])

    # Normalize to procentuals
    representation = representation / representation.sum()

    return representation
