import os
import multiprocessing as mp

import pandas as pd
import numpy as np


def read_data(file_path):
    """This function reads input file and generates a pandas dataframe.
    Parameters
    ----------
    file_path : str
        The file location of input csv

    Returns
    -------
    Pandas.DataFrame
        The dataframe containing the data
    """

    df = pd.read_csv(file_path)

    # columns names correction
    columns_stripped = [el.strip() for el in df.columns]
    df.columns = columns_stripped

    # combining two cols and making datetime col
    df['datetime'] = df['Date'] + df['Time']
    df['datetime'] = pd.to_datetime(
        df['datetime'], format='%Y/%m/%d %H:%M:%S.%f')
    df.set_index('datetime', inplace=True)

    # drop two cols: Date, Time
    df.drop('Date', axis=1, inplace=True)
    df.drop('Time', axis=1, inplace=True)

    # adding close column
    df['Close'] = df['Last']

    # changing column orders
    df = df[['Close'] + list(df.columns[:-1])]
    return df

def get_diff(prices, keep_type=False):
    """
    Calculates the first difference of an input time series.
    
    Parameters
    ----------
    prices (numpy array or or pandas dataframe or pandas series): The time series data for which the first difference needs to be calculated.

    keep_type: (Optional) If set to True, and prices are in the form of a series, then it returns a pandas series.
    
    Returns
    -------
    numpy array or pandas series: An array containing the first difference of the input time series.
    """
    if isinstance(prices, pd.DataFrame):
        columns = [el + '_diff' for el in list(prices.columns)]
    diff_prices = np.diff(prices, axis=0, n=1)
    if (keep_type and isinstance(prices, pd.Series)):
        diff_prices = pd.Series(diff_prices, index=prices.index[1:])
    elif (keep_type and isinstance(prices, pd.DataFrame)):
        diff_prices = pd.DataFrame(diff_prices, index=prices.index[1:], columns=columns)
    return diff_prices

def get_cumulative_mean(prices, keep_type=False):
    """
    Calculates the cumulative mean of an input time series.
    
    Parameters:
    prices (numpy array or pandas series): The time series data for which the cumulative mean needs to be calculated.

    keep_type: (Optional) If set to True, and prices are in the form of a series, then it returns a pandas series.
    
    Returns:
    numpy array or pandas series: An array
    """
    cummean_prices = (np.cumsum(prices) / np.arange(1, len(prices) + 1))
    if (keep_type and isinstance(prices, pd.Series)):
        cummean_prices = pd.Series(cummean_prices, index=prices.index)
    return cummean_prices

def sliding_window(arr, lookback, keep_nrows=False, fillna_value=np.nan):
    """
    Given a 2D array `arr`, return a 3D array containing all windows of size `lookback` along the first axis of `arr`.
    
    Parameters
    ----------
        arr (np.ndarray): 2D input array.
        lookback (int): The size of the windows along the first axis of `arr`.
        keep_nrows (bool): If True, number of rows of arr will bw keeped.
        fillna_value: Value to use fol filling NaN.
    
    Returns
    -------
        np.ndarray: A 3D array with shape `(n_rows - lookback + 1, lookback, n_cols)`, where `n_rows` and `n_cols` are the
        number of rows and columns in `arr`, respectively. if keep_nrows = True, the output is of shape
        `(n_rows, lookback, n_cols)`
    
    Example
    -------
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> sliding_window(arr, 2)
    array([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [7, 8, 9]]])
    """
    n_rows, n_cols = arr.shape
    s = arr.strides[0]
    new_shape = (n_rows - lookback + 1, lookback, n_cols)
    new_strides = (s, s, arr.strides[1])
    out = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    if keep_nrows:
        n_feat = out.shape[-1]
        out = np.concatenate((np.full((lookback - 1, lookback, n_feat), fillna_value), out), axis=0)
    return out
    
def add_white_noise(prices, noise_level, n_samples=1, method='std', keep_type=False):
    """
    This function adds white noise (Gaussian noise) to a given time series of prices.

    Parameters
    ----------
        prices: A 1-dimensional numpy array or pandas series representing the time series of prices.

        noise_level: A scalar value representing the level of noise to add to the time series.

        n_samples: (Optional) Number of samples to generate for each time point in the time series.

        method: (Optional) The method to use for adding noise to the time series. Can be either 'std' or 'cummean'. Default is 'std'.

        keep_type: (Optional) If set to True, and prices are in the form of a series, then it returns a dataframe.

    Returns
    -------
        new_prices: A 2-dimensional numpy array of the same length as prices, representing the time series of prices with added white noise.
    """
    n = len(prices)
    diff_prices = get_diff(prices)
    if method == 'std':
        std_diff = diff_prices.std()
        new_diff_prices = diff_prices.reshape(-1, 1) + noise_level * np.random.normal(0, std_diff, (n-1, n_samples)) 
        new_diff_prices = np.vstack((np.full((1, n_samples), prices[0]), new_diff_prices))
        new_prices = np.cumsum(new_diff_prices, axis=0) 
    elif method == 'cummean':
        diff_prices_cummean = get_cumulative_mean(diff_prices)
        new_diff_prices_cummean = (1 - np.random.normal(0, noise_level, (n-1, n_samples))) * diff_prices_cummean.reshape(-1 , 1)
        new_diff_prices_cumsum = new_diff_prices_cummean * np.arange(1, n).reshape(-1, 1)
        new_diff_prices = np.diff(new_diff_prices_cumsum,axis=0)
        new_diff_prices = np.vstack((new_diff_prices_cummean[0,:], new_diff_prices))
        new_diff_prices = np.vstack((np.full((1, n_samples), prices[0]), new_diff_prices))
        new_prices = np.cumsum(new_diff_prices, axis=0)
    if (keep_type and isinstance(prices, pd.Series)):
        new_prices = pd.DataFrame(new_prices, index=prices.index)
    return new_prices

def convert_to_df(series):
    """
    Convert a pandas Series to a DataFrame with one column.
    
    Parameters
    ----------
        series (pandas series): The series to be converted.
        
    Returns
    -------
        pandas DataFrame: The input series as a DataFrame with one column.
    """
    if isinstance(series, pd.Series):
        series = pd.DataFrame(series.rename(series.name))
    return series

def sync_data(main_file, other_files, fillna=True, keep_timestamp=False):
    """
    Synchronize data from multiple files and merge it into a single dataframe.
    The resulting dataframe will have suffixes added to the columns to indicate
    which file the data came from.
    
    Parameters
    ----------
        main_file (str): The file path of the main file to be used as the base dataframe.
        other_files (list of str): A list of file paths for the files to be merged.
    
    Returns
    -------
        df_sync (pandas DataFrame): The synchronized and merged dataframe.
    """
    df_main = read_data(main_file)
    df_list = []
    for i, file in enumerate(other_files[:]):
        df_list.append(read_data(file))

    df_main = df_main.add_suffix("_main")
    indices = df_main.index

    for i, df in enumerate(df_list):
        df = df.add_suffix(f"_df{i}")
        df_main = pd.merge(df_main, df, left_index=True, right_index=True, how='outer')
        if fillna:
            df_main = df_main.fillna(method='ffill')
        if not keep_timestamp:
            df_main = df_main.loc[df_main.index.isin(indices)]
        duplicate_mask = df_main.index.duplicated(keep='last')
        df_sync = df_main.loc[~duplicate_mask]
    return df_sync

def _sync_data_save(inputs):
    """
    This is an internal function for sync_data_save.

    Parameters
    ----------
        inputs: tuple
        A tuple of inputs where each element is assigned as follows:
        main_file: str: The file path of the main file.
        
        file: str
        The file path of the file that needs to be synchronized with the main file.
        
        fillna: bool
        Whether to fill missing values in the synchronized file.
        
        keep_timestamp: bool
        Whether to keep the timestamp of the original file in the synchronized file.
        
        save_dir: str
        he directory path where the synchronized files will be saved.

    Returns
    -------
        None
    """
    main_file, file, fillna, keep_timestamp, save_dir = inputs
    df = sync_data(main_file, [file,], fillna=fillna, keep_timestamp=keep_timestamp)
    base_name = os.path.basename(file)[:-4]
    save_path = os.path.join(save_dir, f'{base_name}.parquet')
    df.to_parquet(save_path)
    print(f"successfully finished {base_name}")

def sync_data_save(main_file, other_files, save_dir, n_jobs=None, fillna=True, keep_timestamp=False):
    """
    This function is used to save data from multiple files in a synchronized manner.
    The function takes four required parameters and two optional parameters.

    Parameters
    ----------

        main_file: str
        The file path of the main file. This file will be used as the reference file for synchronization.

        other_files: List[str]
        A list of file paths for other files that need to be synchronized with the main file.

        save_dir: str
        The directory path where the synchronized files will be saved.

        n_jobs: int (optional, default=None)
        The number of parallel jobs to run.

        fillna: bool (optional, default=True)
        Whether to fill missing values in the synchronized files.

        keep_timestamp: bool (optional, default=False)
        Whether to keep the timestamp of the original files in the synchronized files.
    
    Returns
    -------
        None
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    other_files = set(other_files)
    other_files = list(other_files.union({main_file}))
    inputs_list = [(main_file, file, fillna, keep_timestamp, save_dir) for file in other_files]
    with mp.Pool(n_jobs) as pool:
        pool.map(_sync_data_save, inputs_list)

def read_parquet(file_path):
    """
    This function reads parquet files and returns a dataframe.

    Parameters
    ----------
    file_path (str): parquet file path.

    Returns
    -------
    pandas dataframe.
    """
    df = pd.read_parquet(file_path)
    return df  