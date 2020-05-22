import pandas as pd
import numpy as np


def MAD_outliers(data: pd.Series) -> pd.Series:
    """Detect outliers use MAD based method and return a binary index

    Detect outliers we can use robust method based on median absolute deviation ( Page 19, Data Cleaning, 2019 ):

    >> ... the median and the median absolute deviation (MAD) that can replace mean and standard deviation, respectively.

    >> The median and MAD lead to a robust outlier detection technique known as Hampel X84 that is 
    >> quite reliable in the face of outliers since it has a breakdown point of 50%. Hampel X84 marks 
    >> outliers as those data points that are more than 1.4826θ MADs away from the median, where θ is 
    >> the number of standard deviations away from the mean one would have used if there were no 
    >> outliers in the dataset. The constant 1.4826 is derived under a normal distribution, where one 
    >> standard deviation away from the mean is about 1.4826 MADs.
    
    """
    num_std = 3
    theta = 1.4826

    median = data.median()
    mad = np.median((data - median).abs())

    lower = median - mad * (theta * num_std)
    upper = median + mad * (theta * num_std)

    return (data < lower) | (data > upper)


def date_counts(
    data: pd.DataFrame, date_column: str = "datetime", date_index=False
) -> pd.DataFrame:
    """
    Counts the number rows per date and returns a dataframe with
    counts stored in column "total". Depending on "date_index" date stored
    either as index or as a separate column in the new data frame
    """

    agg_data = data.groupby(data[date_column].dt.date)[date_column].count()
    agg_data = agg_data.to_frame("total")

    if not date_index:
        agg_data = agg_data.reset_index()

    return agg_data
