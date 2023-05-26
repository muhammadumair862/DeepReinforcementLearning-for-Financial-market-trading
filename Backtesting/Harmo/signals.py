from datetime import datetime
from itertools import product, islice
from multiprocessing import Pool
import psutil
import gc
import time

import numpy as np
import pandas as pd
from numba import njit


@njit
def _bar_based_exit_internal(_indices, in_position, entries_arr, new_entries, entry_prices, price_arr, count_bars, num_bars, exits, exit_prices):
    """This is an internal function for function bar_based_exit.
    
    Parameters
    ----------
    _indices : DateTimeIndex
        Indices of price series.
    
    in_position : numpy.ndarray
        Whether a long position is already open or not.
    
    entries_arr : numpy.ndarray
        Data of entries dataframe.

    new_entries : numpy.ndarray
        New entries array.
    
    entry_prices : numpy.ndarray
        Prices for entries.
    
    price_arr : numpy.ndarray
        Data of price dataframe.
    
    count_bars : numpy.ndarray
        The array that counts number of bars.
    
    num_bars : int
        cutoff value for number of bars.
    
    exits : numpy.ndarray
        An array of exit signals.
    
    exit_prices : numpy.ndarray
        An array of exit prices.

    Returns
    -------
    new_entries : numpy.ndarray(bool)
        entry signals array.
    
    exits : numpy.ndarray(bool)
        exit signal array.
    
    entry_prices : numpy.ndarray(float)
        entry prices array.

    exit_prices : numpy.ndarray
        exit prices array.
    """

    for i in _indices:
        # If there is an entry signal and the trader is not already in a long position.
        mask1 = (~in_position & entries_arr[i])
        new_entries[i][mask1] = True
        entry_prices[i][mask1] = price_arr[i]
        count_bars[mask1] = 0
        in_position[mask1] = True

        # If the trader is currently holding a long position and the cutoff value for the number of bars has just been reached.
        mask2 = (in_position & (count_bars == num_bars))
        exits[i][mask2] = True
        exit_prices[i][mask2] = price_arr[i]
        count_bars[mask2] = 0
        in_position[mask2] = False

        # otherwise
        mask3 = (~mask1 & ~mask2)
        count_bars[mask3] += 1

    return new_entries, exits, entry_prices, exit_prices

def bar_based_exit(prices=None, entries=None, num_bars=10):
    """This function implements an exit strategy based on the number of bars.
    
    Parameters
    ----------
    prices : Pandas.Series
        The prices series.
    
    entries : Pandas.DataFrame or Pandas.Series
        The entries point signaled by a strategy.
    
    num_bars : int
        The cutoff for number of bars.

    Returns
    -------
    new_entries : Pandas.DataFrame
        The new entries signal dataframe.
    
    exits : Pandas.DataFrame
        Exits signal dataframe.
    
    entry_prices : Pandas.DataFrame
        Entry prices dataframe.

    exit_prices : Pandas.DataFrame
        Exit prices dataframe.
    """
    if isinstance(entries, pd.Series):
        entries = pd.DataFrame(entries, index=entries.index)
    entries_arr = entries.to_numpy()
    price_arr = prices.to_numpy()
    __indices = entries.index
    __columns = entries.columns
    _shape = entries_arr.shape
    _indices = np.arange(_shape[0])
    entry_prices = np.zeros(_shape, dtype=float)
    exit_prices = np.zeros(_shape, dtype=float)
    new_entries = np.full(_shape, False)
    exits = np.full(_shape, False)
    in_position = np.full(_shape[1:], False)
    count_bars = np.zeros(_shape[1:], dtype=int)
    new_entries, exits, entry_prices, exit_prices = _bar_based_exit_internal(_indices, in_position, entries_arr, new_entries, entry_prices, price_arr, count_bars, num_bars, exits, exit_prices)

    # creating output dataframes
    new_entries = pd.DataFrame(new_entries, index=__indices, columns=__columns)
    exits = pd.DataFrame(exits, index=__indices, columns=__columns)
    entry_prices = pd.DataFrame(entry_prices, index=__indices, columns=__columns)
    exit_prices = pd.DataFrame(exit_prices, index=__indices, columns=__columns)

    # exiting open positions at the end 
    open_columns = (new_entries.sum(axis=0) > exits.sum(axis=0))
    last_datetime = __indices[-1]
    exits.loc[last_datetime, open_columns] = True
    exit_prices.loc[last_datetime, open_columns] = prices.loc[last_datetime]

    return new_entries, exits, entry_prices, exit_prices

@njit
def _take_profit(price_arr, counter, sl_price, RR=1):
    """This function calculates take profit prices.
    
    Parameters
    ----------
    price_arr : numpy.ndarray
        Array of prices.
    
    counter :  int
        Dataframe index.
    
    sl_price : np.ndarray (float)
        An array of stop loss prices.
    
    RR : float
        Risk reward ratio.
    
    Returns
    -------
    np.ndarray (float)
        An array of take profit prices.
    """
    sl = np.abs(price_arr[counter] - sl_price)
    tp = RR * sl
    tp_price = tp + price_arr[counter]
    return tp_price

@njit
def _bar_range(price_arr, low_arr, high_arr, counter):
    """This function indicates whether prices are within the range of low and high prices or not.

    Parameters
    ----------
    price_arr : numpy.ndarray (float)
        Array of prices.

    low_arr : numpy.ndarray (float)
        Array of low prices.

    high_arr : numpy.ndarray (float)
        Array of high prices.
    
    counter : int
        Dataframe index.

    Returns
    -------
    numpy.ndarray (bool)
        An array that indicates whether prices are whitin range or not.
    """
    return (price_arr <= high_arr[counter]) & (price_arr >= low_arr[counter])

@njit
def _in_uptrend(open_arr, counter):
    """This function indicates whether open prices are increasing or not.

    Parameters
    ----------
    open_arr : numpy.ndarray (float)
        Array of open prices.
    
    counter : int
        Dataframe index.

    Returns
    -------
    numpy.ndarray (bool) 
        An array indicating whther the open price is increasing or not.
    """
        
    return np.where(np.array(counter==0), np.full(open_arr.shape[1:], False), np.where(open_arr[counter] > open_arr[counter - 1], True, False))

@njit      
def _in_downtrend(open_arr, counter):
    """This function indicates whether open prices are declining or not.

    Parameters
    ----------
    open_arr : numpy.ndarray (float)
        Array of open prices.
    
    counter : int
        Dataframe index.

    Returns
    -------
    numpy.ndarray (bool) 
        An array indicating whther the open price is declining or not.
    """
    return np.where(np.array(counter==0), np.full(open_arr.shape[1:], False), np.where(open_arr[counter] < open_arr[counter - 1], True, False))

@njit
def _fixed_sl_exit_internal(_indices, entries_arr, in_position, sl_price, close_arr, SL, tp_price, RR, new_entries, low_arr, high_arr, open_arr, prices, exits, _take_profit_func, _bar_range_func, _in_uptrend_func, _in_downtrend_func):
    """This is an internal function for function fixed_sl_exit

    Parameters
    ----------
    _indices : numpy.ndarray (int)
        Indices of dataframe.

    entries_arr : numpy.ndarray (bool)
        Array of entry points.
    
    in_position : numpy.ndarray (bool)
        Is the trader holding a long position.
    
    sl_price : numpy.ndarray (float)
        Array of stop loss prices.
    
    close_arr : numpy.ndarray
        Array of close prices.
    
    SL : float
        Fixed stop loss ratio.
    
    tp_price : np.ndarray (float)
        Take profit prices array.
    
    RR : float
        Risk reward ratio.
    
    new_entries : np.ndarray (bool)
        Array of new entries.

    low_arr : np.ndarray (float)
        Array of low prices.

    high_arr : np.ndarray (float)
        Array of high prices.
    
    open_arr : np.ndarray (float)
        Array of open prices.
    
    prices : np.ndarray (float)
        Array of prices.
    
    exits : np.ndarray (bool)
        Array of sxit signals.
    
    _take_profit_func : numba.core.registry.CPUDispatcher
        Take profit function.

    _bar_range_func : numba.core.registry.CPUDispatcher
        bar range function.
    
    _in_uptrend_func : numba.core.registry.CPUDispatcher
        Uptrend function.
    
    _in_downtrend_func : numba.core.registry.CPUDispatcher
        Downtrend function.

    Returns
    -------
    exits : numpy.ndarray (bool)
        Array of exit signals.
    
    prices : numpy.ndarray (float)
        Array of prices.
    
    new_entries : numpy.ndarray (bool)
        Array of new entry signals.
    """

    for i in _indices:
        # If there is an entry signal and the trader is not in a long position.
        mask1 = (entries_arr[i] & ~in_position)
        sl_price[mask1] = close_arr[i] - SL
        tp_price[mask1] = _take_profit(close_arr, i, sl_price, RR=RR)[mask1]
        in_position[mask1] = True
        new_entries[i][mask1] = True 

        bar_range_tp = _bar_range(tp_price, low_arr, high_arr, i)
        bar_range_sl = _bar_range(sl_price, low_arr, high_arr, i)

        # If the trader is in a long position.
        mask2 = in_position
        
        # If the trader is in a long position and both the take profit price and the stop loss prices are within the bar range
        mask2_1 = mask2 & bar_range_tp & bar_range_sl

        # If the open price is increasing, the trader is in a long position, and both the take profit price and stop loss prices are within the bar range
        mask2_1_1 = mask2_1 & _in_uptrend(open_arr, i)
        prices[i][mask2_1_1] = tp_price[mask2_1_1]
        exits[i][mask2_1_1] = True
        in_position[mask2_1_1] = False

        # If the open price is decreasing, the trader is in a long position, and both the take profit price and stop loss prices are within the bar range
        mask2_1_2 = mask2_1 & _in_downtrend(open_arr, i)
        prices[i][mask2_1_2] = sl_price[mask2_1_2]
        exits[i][mask2_1_2] = True
        in_position[mask2_1_2] = False

        # If the trader is in a long position and only the take profit price is within the bar range
        mask2_2 = mask2 & bar_range_tp & ~bar_range_sl
        prices[i][mask2_2] = tp_price[mask2_2]
        exits[i][mask2_2] = True
        in_position[mask2_2] = False

        # If the trader is in a long position and only the stop loss price is within the bar range
        mask2_3 = mask2 & ~bar_range_tp & bar_range_sl
        prices[i][mask2_3] = sl_price[mask2_3]
        exits[i][mask2_3] = True
        in_position[mask2_3] = False

        # If neither the take profit price nor the stop loss prices are within the bar range and the trader is in a long position
        mask2_4 = mask2 & ~bar_range_tp & ~bar_range_sl

        # If the open price is greater than the take profit price, the trader is in a long position, and neither the take profit price nor the stop loss prices are within the bar range
        mask2_4_1 = mask2_4 & (open_arr[i] > tp_price)
        prices[i][mask2_4_1] = open_arr[i]
        exits[i][mask2_4_1] = True
        in_position[mask2_4_1] = False

        # If the open price is less than the stop loss price, the trader is in a long position, and neither the take profit price nor the stop loss prices are within the bar range
        mask2_4_2 = mask2_4 & (sl_price > open_arr[i])
        prices[i][mask2_4_2] = open_arr[i]
        exits[i][mask2_4_2] = True
        in_position[mask2_4_2] = False
    return exits, prices, new_entries

def fixed_sl_exit(df=None, entries=None, RR=2, SL=3):
    """This function calculates exit signals using a fixed stop loss parameter.

    Parameters
    ----------    
    df : Pandas.DataFrame
        Dataset dataframe.
    
    entries : Pandas.DataFrame or Pandas.Series (bool)
        The entries point signaled by a strategy.
    
    RR : float
        The risk-reward ratio.

    SL : float
        The stop loss ratio.

    Returns
    -------
    new_entries : Pandas.DataFrame (bool)
        New entry points.
        
    exits : Pandas.DataFrame (bool)
        Exit points.
    
    entry_prices : Pandas.DataFrame
        Entry prices.
    
    prices : Pandas.DataFrame
        prices.
    """
    if isinstance(entries, pd.Series):
        entries = pd.DataFrame(entries, index=entries.index)
    entries_arr = entries.to_numpy()
    close_arr = df['Close'].to_numpy()
    open_arr = df['Open'].to_numpy()
    high_arr = df['High'].to_numpy()
    low_arr = df['Low'].to_numpy()

    _shape = entries_arr.shape
    _indices = np.arange(_shape[0])
    __indices = entries.index
    __columns = entries.columns
    prices = np.zeros(_shape, dtype=float)
    new_entries = np.full(_shape, False)
    in_position = np.full(_shape[1:], False)
    sl_price = np.zeros(_shape[1:], dtype=float)
    tp_price = np.zeros(_shape[1:], dtype=float)
    exits = np.full(_shape, False)
    exits, prices, new_entries = _fixed_sl_exit_internal(_indices, entries_arr, in_position, sl_price, close_arr, SL, tp_price, RR, new_entries, low_arr, high_arr, open_arr, prices, exits, _take_profit, _bar_range, _in_uptrend, _in_downtrend)
    exits = pd.DataFrame(exits, index=__indices, columns=__columns)
    prices = pd.DataFrame(prices, index=__indices, columns=__columns)
    new_entries =  pd.DataFrame(new_entries, index=__indices, columns=__columns)
    entry_prices = df['Close']
    # exiting open positions at the end 
    open_columns = (new_entries.sum(axis=0) > exits.sum(axis=0))
    last_datetime = __indices[-1]
    exits.loc[last_datetime, open_columns] = True
    return new_entries, exits, entry_prices, prices

@njit
def _time_based_exit_internal(timestamps_arr, entries_arr, in_position, exit_time, tdelta, new_entries, entries_prices, price_arr, exits, exit_prices, dafault_datetime):
    """This is an internal function for function time_based_exit.
    
    Parameters
    ----------
    timestamps_arr : numpy.ndarray (datetime)
        datetime indices.
    
    entries_arr : numpy.ndarray (bool)
        Array of entry points.
    
    in_position : numpy.ndarray (bool)
        Is the trader holding a long position.
    
    exit_time : numpy.ndarray (datetime)
        Array of exit times.
    
    tdelta : timedelta
        The amount of time for holding a long position.    

    new_entries : numpy.ndarray (bool)
        New entries array.    
    
    entries_prices : numpy.ndarray (float)
        Array of entries prices.    
    
    price_arr : numpy.ndarray (float)
        Array of prices.
    
    exits : numpy.ndarray (bool)
        Array of exits.
    
    exit_prices : numpy.ndarray (float)
        Array of exit prices.
    
    dafault_datetime : datetime
        Default datetime object. 

    Returns
    -------

    new_entries : numpy.ndarray (bool)
        New entries array.        
    
    exits : numpy.ndarray (bool)
        Exits array.
    
    entries_prices : numpy.ndarray (float)
        Array of prices.
    
    exit_prices : numpy.ndarray (float)
        Array of exit prices.
    """

    for i, timestamp in enumerate(timestamps_arr):
        mask1 = entries_arr[i] & ~in_position 
        
        exit_time[mask1] = (timestamp + tdelta)
        new_entries[i][mask1]  = True
        entries_prices[i][mask1] = price_arr[i]
        in_position[mask1] = True  

        mask2 = in_position & (exit_time <= timestamp)
        exit_time[mask2] = dafault_datetime
        in_position[mask2] = False
        exits[i][mask2] = True
        exit_prices[i][mask2] = price_arr[i]
    return new_entries, exits, entries_prices, exit_prices

def time_based_exit(prices, entries, hours=0, minutes=0, seconds=0):
    """This function implements a time-based exit strategy.
    
    Parameters
    ----------

    prices : numpy.ndarray (float)
        Array of prices.
    
    entries : pandas.DataFrame or pandas.Series (bool)
        Array of entries.
    
    hours : int
        The number of hours for holding a long position.
    
    minutes : int
        The number of minutes for holding a long position.
        
    seconds : int
        The number of seconds for holding a long position.

    Returns
    -------

    new_entries :  pandas.DataFrame (bool)
        The new entries signal dataframe.
    
    exits : pandas.DataFrame (bool)
        Exits signal dataframe.
    
    entries_prices : pandas.DataFrame (float)
        Entry prices dataframe.
    
    exit_prices : pandas.DataFrame (flaot)
        Exit prices dataframe.
    """
    if isinstance(entries, pd.Series):
        entries = pd.DataFrame(entries, index=entries.index)
    prices_arr = prices.to_numpy()
    __indices = prices.index
    _timestamps_arr = np.array(prices.index, dtype='datetime64[s]')
    _columns = entries.columns
    _shape = entries.shape
     
    entries_arr = entries.to_numpy()
    entries_prices = np.zeros(_shape, dtype=float)
    exit_prices = np.zeros(_shape, dtype=float)
    new_entries = np.full(_shape, False)
    exits = np.full(_shape, False)
    in_position = np.full(_shape[1:], False) 
    dafault_datetime = np.datetime64(datetime(1970, 1, 1, 0, 0, 0), 's')
    exit_time = np.full(_shape[1:], dafault_datetime)
    tdelta = np.timedelta64(hours * 3600 + minutes * 60 + seconds, 's')
    
    new_entries, exits, entries_prices, exit_prices =  _time_based_exit_internal(_timestamps_arr, entries_arr, in_position, exit_time, tdelta, new_entries, entries_prices, prices_arr, exits, exit_prices, dafault_datetime)
    new_entries = pd.DataFrame(new_entries, index=__indices, columns=_columns)
    exits = pd.DataFrame(exits, index=__indices, columns=_columns)
    entries_prices = pd.DataFrame(entries_prices, index=__indices, columns=_columns)
    exit_prices = pd.DataFrame(exit_prices, index=__indices, columns=_columns)

    # exiting open positions at the end 
    open_columns = (new_entries.sum(axis=0) > exits.sum(axis=0))
    last_datetime = __indices[-1]
    exits.loc[last_datetime, open_columns] = True
    exit_prices.loc[last_datetime, open_columns] = prices.loc[last_datetime]
    return new_entries, exits, entries_prices, exit_prices

def generate_cartesian_product(cols, chunk_size):
    """
    Generates cartesian product of the columns in 'cols' and yields the result in chunks of size 'chunk_size'
    
    Parameters
    ----------    
    cols : list of lists
        List of columns whose cartesian product is to be generated.

    chunk_size : int
        Number of items in each chunk of the cartesian product.
    
    Yields
    ------
    list : A chunk of the cartesian product of the columns in 'cols'.
    """
    cartesian_product = product(*cols)
    while True:
        chunk = list(islice(cartesian_product, chunk_size))
        if not chunk:
            break
        yield list(zip(*chunk))

def parallel_processing(func, cols, chunk_size, n_cores):
    """
    Applies the function 'func' to the cartesian product of the columns in 'cols' in parallel using 'n_cores' number of cores.
    
    Parameters
    ----------
    func : function
        The function that is to be applied to each element of the cartesian product.
    cols : list of lists
        List of columns whose cartesian product is to be generated.
    chunk_size : int
        Number of items in each chunk of the cartesian product.
    n_cores : int
        Number of cores to use for parallel processing.
    """
    with Pool(processes=n_cores) as pool:
        try:
            for result in pool.imap(func, generate_cartesian_product(cols, chunk_size)):
                gc.collect()
        except MemoryError:
            print('ERROR: Commit memory is filled. reduce chunk size.')
            raise            
        