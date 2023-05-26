import pandas as pd
from scipy import stats
import numpy as np
from datetime import timedelta
import vectorbt as vbt
import talib
from datetime import datetime, time
from Harmo.data import *
from Harmo.signals import *


# Last Bars base SL Exit Strategy
def exit_signal_1(df=None, entries=None, rr=1, Num_Bars=10):
    entries, exits, entry_price, exit_price = bar_based_exit(df.Close, entries, num_bars=Num_Bars)  
    df1 = pd.DataFrame({})
    df1['Entry'] = entries
    df1['Exit'] = exits
    df1['Entry Price'] = entry_price
    df1['Exit Price'] = exit_price
    df1['Trade Compeletion Days'] = 0
    df1['Trade Status (Win/Loss)'] = "Nothing"
    
    exit_len = len(df1['Exit'][df1['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df1['Trade Compeletion Days'][df1['Exit'] == True] = df1['Exit'][df1['Exit'] == True].index - df1['Entry'][df1['Entry'] == True][:exit_len].index
    df1['Trade Status (Win/Loss)'][df1['Exit'] == True] = np.where((df1['Exit Price'][df1['Exit']==True].reset_index(drop=True) - 
                                                                    df1['Entry Price'][df1['Entry']==True].iloc[:exit_len].reset_index(drop=True))>0, 'Win', 'Loss' )
    df1.index.name = 'Open time'

    return df1

# Last Bars base SL Exit Strategy
def exit_signal(df=None, entries=None, rr=1, Num_Bars=10):
    
    """
        Calculate exit signals by calculating stoploss from last x num of bars

        Args :
            df (Dataframe): The dataframe of dataset
            entries (Sieres): Entries signals sieres 
            rr (float): risk/reward ratio
            num_bar (int): The last x number of bars from entry signal point

        Return
            df (dataframe): The dataframe having entries & exits signals, days to compelete
                            each trade and status of each trade. 
    """

    df['Entry'] = False
    df['Entry Price'] = df['Close']
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"
    
    # calculate stoploss, take profit & new entry points 
    df['SL Price'] = np.where(entries == True, df['Low'].rolling(Num_Bars).min(), 0)
    df['TP Price'] = np.where(df['SL Price']>0, (df['Close'] - df['SL Price']).abs() * rr + df['Close'], 0)
    df['Entry Price'] = np.where(entries==True, df['Close'], 0)

    # calculate trade status(win/loss) after adding fee 
    # df['Trade Status (Win/Loss)'] = np.where(((df['Exit Price'][df['Exit'] == True] - df['Entry Price'][df['Entry'] == True][:exit_len])-Fee)>=0,'Win',
    #                                     np.where(((df['Exit Price'][df['Exit'] == True] - df['Entry Price'][df['Entry'] == True][:exit_len])-Fee)<0,'Loss',"Nothing"))

    status = True

    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            status = False
            SL_Price=df['SL Price'].iloc[i]
            TP_Price=df['TP Price'].iloc[i]

            df['Entry'].iloc[i] = True

        elif status == False:
            
            if ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):

                if (df['Open'].iloc[i] < df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = TP_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True
                elif (df['Open'].iloc[i] > df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = SL_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True
            
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = TP_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                status=True

            elif ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                  ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = SL_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                status=True

             
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                if df['Open'].iloc[i]>TP_Price:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True

                elif SL_Price>df['Open'].iloc[i]:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True            

    exit_len = len(df['Exit'][df['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df['Trade Compeletion Days'][df['Exit'] == True] = df['Exit'][df['Exit'] == True].index - df['Entry'][df['Entry'] == True][:exit_len].index
    return df


#  $$$$$$$$$  Fixed SL amount with rr Exit Strategy
def fixed_exit_signal_1(df=None,entries=None,RR=1,SL=None):
    entries, exits, entry_price, exit_price = fixed_sl_exit(df=df, entries=entries, RR=RR, SL=SL)
    df1 = pd.DataFrame({})
    df1['Entry'] = entries
    df1['Exit'] = exits
    df1['Entry Price'] = entry_price
    df1['Exit Price'] = exit_price
    df1['Trade Compeletion Days'] = 0
    df1['Trade Status (Win/Loss)'] = "Nothing"
    
    exit_len = len(df1['Exit'][df1['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df1['Trade Compeletion Days'][df1['Exit'] == True] = df1['Exit'][df1['Exit'] == True].index - df1['Entry'][df1['Entry'] == True][:exit_len].index
    df1['Trade Status (Win/Loss)'][df1['Exit'] == True] = np.where((df1['Exit Price'][df1['Exit']==True].reset_index(drop=True) - 
                                                                    df1['Entry Price'][df1['Entry']==True].iloc[:exit_len].reset_index(drop=True))>0, 'Win', 'Loss' )
    df1.index.name = 'Open time'

    return df1


#  $$$$$$$$$  Fixed SL amount with rr Exit Strategy
def fixed_exit_signal(df=None,entries=None,RR=1,SL=None):

    """
        calculate exit signals on the basis of fixed stoploss and risk reward ratio 

        Args :
            df (Dataframe): The dataframe of data
            entries (Sieres): Entries signals sieres
            RR (float): risk reward ratio
            SL (float): stoploss amount  

        Return
            entries1 (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            prices (Series float64): exit signals prices
    """
    df['Entry'] = False
    df['Entry Price'] = df['Close']
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"
    
    # calculate stoploss, take profit & new entry points 
    df['SL Price'] = np.where(entries == True, df['Close']-SL, 0)
    df['TP Price'] = np.where(df['SL Price']>0, SL * RR + df['Close'], 0)
    df['Entry Price'] = np.where(entries==True, df['Close'], 0)

    status = True

    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            status = False
            SL_Price=df['SL Price'].iloc[i]
            TP_Price=df['TP Price'].iloc[i]

            df['Entry'].iloc[i] = True

        elif status == False:
            
            if ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):

                if (df['Open'].iloc[i] < df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = TP_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True
                elif (df['Open'].iloc[i] > df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = SL_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True
            
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = TP_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                status=True

            elif ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                  ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = SL_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                status=True

             
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                if df['Open'].iloc[i]>TP_Price:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True

                elif SL_Price>df['Open'].iloc[i]:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True            

    exit_len = len(df['Exit'][df['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df['Trade Compeletion Days'][df['Exit'] == True] = df['Exit'][df['Exit'] == True].index - df['Entry'][df['Entry'] == True][:exit_len].index
    return df


# Function to calculate exit signals on the number of bars
# -----------------------------  Start  -----------------------------
def bar_base_exit_1(df = None, entries = None, num_bars = None):
    entries, exits, entry_price, exit_price = bar_based_exit(df.Close, entries, num_bars=num_bars) 
    df1 = pd.DataFrame({})
    df1['Entry'] = entries
    df1['Exit'] = exits
    df1['Entry Price'] = entry_price
    df1['Exit Price'] = exit_price
    df1['Trade Compeletion Days'] = 0
    df1['Trade Status (Win/Loss)'] = "Nothing"
    
    exit_len = len(df1['Exit'][df1['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df1['Trade Compeletion Days'][df1['Exit'] == True] = df1['Exit'][df1['Exit'] == True].index - df1['Entry'][df1['Entry'] == True][:exit_len].index
    df1['Trade Status (Win/Loss)'][df1['Exit'] == True] = np.where((df1['Exit Price'][df1['Exit']==True].reset_index(drop=True) - 
                                                                    df1['Entry Price'][df1['Entry']==True].iloc[:exit_len].reset_index(drop=True))>0, 'Win', 'Loss' )
    df1.index.name = 'Open time'

    return df1


# Function to calculate exit signals on the number of bars
# -----------------------------  Start  -----------------------------
def bar_base_exit(df = None, entries = None, num_bars = None):
    
    """
        calculate exit signals with x num of bars 

        Args :
            price (Series): The Series of closing or other price
            entries (Sieres): Entries signals sieres
            num_bars (float): num of bars

        Return
            new_entries (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            exit_prices (Series float64): exit signals prices
    """
    df['Entry'] = False
    df['Entry Price'] = 0
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"

    status=True
    count_bars=0
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            count_bars=1
            status=False
            df['Entry'].iloc[i] = True

        
        elif status==False and count_bars==num_bars:
            status=True
            df['Exit Price'].iloc[i] = df['Close'].iloc[i]
            df['Exit'].iloc[i] = True
            df['Trade Status (Win/Loss)'].iloc[i] = np.where((df['Exit Price'].iloc[i]-df['Entry Price'].iloc[i-num_bars])>0,"Win","Loss")
            count_bars=1
            
        else:
            count_bars+=1

    df['Entry Price'] = np.where((df['Entry']==True),df['Close'],0)
    return df

# -----------------------------  End  -----------------------------


# Function to calculate exit signals on the basis of time
# -----------------------------  Start  -----------------------------
def time_base_exit_1(df = None, entries = None, hour = None,
                        min = None, sec = None, trade_type = None):
    entries, exits, entry_price, exit_price = time_based_exit(df.Close, entries, hours=hour,
         minutes=min, seconds=sec)
    df1 = pd.DataFrame({})
    df1['Entry'] = entries
    df1['Exit'] = exits
    df1['Entry Price'] = entry_price
    df1['Exit Price'] = exit_price
    df1['Trade Compeletion Days'] = 0
    df1['Trade Status (Win/Loss)'] = "Nothing"
    
    exit_len = len(df1['Exit'][df1['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df1['Trade Compeletion Days'][df1['Exit'] == True] = df1['Exit'][df1['Exit'] == True].index - df1['Entry'][df1['Entry'] == True][:exit_len].index
    df1['Trade Status (Win/Loss)'][df1['Exit'] == True] = np.where((df1['Exit Price'][df1['Exit']==True].reset_index(drop=True) - 
                                                                    df1['Entry Price'][df1['Entry']==True].iloc[:exit_len].reset_index(drop=True))>0, 'Win', 'Loss' )

    df1.index.name = 'Open time'

    return df1


# Function to calculate exit signals on the basis of time
# -----------------------------  Start  -----------------------------
def time_base_exit(df = None, entries = None, hour = None,
                        min = None, sec = None, trade_type = None):
    """
        calculate exit signals with time constraint (exit after specific hour, minute or second)

        Args :
            df (Dataframe): The dataframe of data
            entries (Sieres): Entries signals sieres
            hour (int): num of hour
            min (int): num of minutes
            sec (int): num of seconds

        Return
            new_entries (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            exit_prices (Series float64): exit signals prices
    """

    df['Entry'] = False
    df['Entry Price'] = 0
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"

    status=True
    Ex_time=0
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in entries.index:
        # Find Exit Prices of Enteries Signals
        if entries.loc[i] and status:
            if hour:
                Ex_time=i+timedelta(hours=hour)
            elif min:
                Ex_time=i+timedelta(minutes=min)
            elif sec:
                Ex_time=i+timedelta(seconds=sec)

            df['Entry'].loc[i] = True
            df['Entry Price'].loc[i] = df['Close'].loc[i]            
            status=False
        
        elif status==False and Ex_time<=i:
            Ex_time=0
            status=True
            df['Exit'].loc[i] = True
            df['Exit Price'].loc[i] = df['Close'].loc[i]
    
    return df

# -----------------------------  End  -----------------------------


# Function to calculate exit signals on the basis of time
# -----------------------------  Start  -----------------------------
def read_file(path=None, num_last_bars=0, standard_format=False):
    
    """
        Convert the pandas dataframe into vbt dataframe format after format it

        Args :
            path (str): The path of csv or text file
            num_last_bars (int): The number of rows/bars wants to get from last of file

        Return :
            data (dataframe): The vbt or formated dataframe  
    """

    # load data
    try:
        data = pd.read_csv(path)
    except:
        print("Incorrect File Path or wrong File")
        return None

    # preprocess columns names and create datetime index
    try:
        data['Open time'] = pd.to_datetime(data['Date'] + data[' Time'])
        data.index = data['Open time']
        data.columns = [i.strip() for i in data.columns] 
        data.rename(columns={'Last':'Close'},inplace=True)
        data.drop(columns=['Open time','Date','Time'], inplace=True)
        if standard_format:
            try:
                data = data[['Open','High','Low','Close','Volume','NumberOfTrades', 'BidVolume', 'AskVolume']]
            except:
                data = data[['Open','High','Low','Close','Volume','# of Trades', 'Bid Volume', 'Ask Volume']]
    except:
        print("Columns not same as sierra chart file")
        return None

    # select rows or bars from end of data
    return data[-(num_last_bars):]
    

# -----------------------------  End  -----------------------------


#  ^^^^^^^^^^^^^^^  Entries Signal Logic  ^^^^^^^^^^^^^^^^^
def macd_logic(close_price=None, volume=None, fast=6,
              slow=22, macd_ma_period=6, ema_period=22,
              ema_vol_period=8):
    macd = vbt.MACD.run(close_price,fast,slow,fast)
    ma_macd = vbt.MA.run(macd.macd,macd_ma_period)
    ema = vbt.talib("EMA").run(close_price,ema_period).real.to_numpy()
    ema_vol = vbt.talib("EMA").run(volume,ema_vol_period).real.to_numpy()

    entries = np.where(macd.macd_crossed_above(ma_macd.ma).to_numpy() & (close_price>ema) & 
                            (volume>ema_vol), True, False)
    return entries

def x_bar_entry(close_price, x_bar=10):
    idx = close_price.reset_index().index
    entries = np.where(idx%x_bar==0, True, False)
    entries = pd.Series(entries, index=close_price.index)
    return entries

def macd_entry(close_price, volume):
    ind = vbt.IndicatorFactory(
        class_name = "MACD_EMA_EMAVol_logic",
        short_name = "macde_emavol",
        input_names  = ["close_price", "volume"],
        param_names = ["fast", "slow", "macd_ma_period", "ema_period", "ema_vol_period"],
        output_names = ["entries"],
            ).with_apply_func(
                macd_logic, 
                keep_pd=True, 
                takes_1d=True
    )
    res = ind.run(
        close_price=close_price,
        volume=volume,
        fast=[6],
        slow=36,
        macd_ma_period=14, 
        ema_period=76,
        ema_vol_period=11,
        param_product = True
        )
    return res.entries


# $$$$$$$$$ short entry logic $$$$$$$$$
# Long Call
def long(close):
    macd = vbt.talib("MACD").run(close)
    rsi = vbt.talib("RSI").run(close)
    # entries = np.where(macd.macd_crossed_above(macd.macdsignal) & (rsi.real.shift(1)<=40 | rsi.real.shift(2)<=40 | rsi.real.shift(3)<=40 | rsi.real.shift(4)<=40 | rsi.real.shift(5)<=40), True, False)
    entries = np.where((macd.macd_crossed_above(macd.macdsignal)) & (rsi.real.shift(1) <= 40).astype(bool) | (rsi.real.shift(2) <= 40).astype(bool) | (rsi.real.shift(3) <= 40).astype(bool) | (rsi.real.shift(4) <= 40).astype(bool) | (rsi.real.shift(5) <= 40).astype(bool), True, False)
    return  entries

def buy_indicator(close_price):
    ind = vbt.IndicatorFactory(
    class_name = "Combination_Trend_Long",
    short_name = "long",
    input_names  = ["close"],
    output_names = [ "entries"],
        ).with_apply_func(
            # trend_chaser, 
            long,
            keep_pd=True, 
            takes_1d=True
    )
    res = ind.run(
    close = close_price,
    # param_product = True 
    )
    return res.entries

# Short Call
def short(close):
    macd = vbt.talib("MACD").run(close)
    rsi = vbt.talib("RSI").run(close)
    # entries = np.where(macd.macd_crossed_above(macd.macdsignal) & (rsi.real.shift(1)<=40 | rsi.real.shift(2)<=40 | rsi.real.shift(3)<=40 | rsi.real.shift(4)<=40 | rsi.real.shift(5)<=40), True, False)
    entries = np.where((macd.macdsignal_crossed_above(macd.macd)) & (rsi.real.shift(1) >= 65).astype(bool) | (rsi.real.shift(2) >= 65).astype(bool) | (rsi.real.shift(3) >= 65).astype(bool) | (rsi.real.shift(4) >= 65).astype(bool) | (rsi.real.shift(5) >= 65).astype(bool), True, False)
    return  entries

def sell_indicator(close_price):
    ind = vbt.IndicatorFactory(
    class_name = "Combination_Trend_Long",
    short_name = "short",
    input_names  = ["close"],
    output_names = [ "entries"],
        ).with_apply_func(
            # trend_chaser, 
            short,
            keep_pd=True, 
            takes_1d=True
    )
    res = ind.run(
    close = close_price,
    # param_product = True 
    )
    return res.entries


#  $$$$$$$$$  Fixed SL amount with rr Exit Strategy (For Short Signal)
def fixed_exit_shortsignal(df=None,entries=None,RR=1,SL=None):

    """
        calculate exit signals on the basis of fixed stoploss and risk reward ratio 

        Args :
            df (Dataframe): The dataframe of data
            entries (Sieres): Entries signals sieres
            RR (float): risk reward ratio
            SL (float): stoploss amount  

        Return
            entries1 (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            prices (Series float64): exit signals prices
    """
    df['Entry'] = False
    df['Entry Price'] = df['Close']
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"
    
    # calculate stoploss, take profit & new entry points 
    df['SL Price'] = np.where(entries == True, df['Close'] + SL, 0)
    df['TP Price'] = np.where(df['SL Price']>0, df['Close'] - SL * RR, 0)
    df['Entry Price'] = np.where(entries==True, df['Close'], 0)

    status = True

    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            status = False
            SL_Price = df['SL Price'].iloc[i]
            TP_Price = df['TP Price'].iloc[i]

            df['Entry'].iloc[i] = True

        elif status == False:
            
            if ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):

                if (df['Open'].iloc[i] < df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = SL_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True
                elif (df['Open'].iloc[i] > df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = TP_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True
            
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = TP_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                status=True

            elif ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                  ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = SL_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                status=True

             
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                if df['Open'].iloc[i]<TP_Price:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True

                elif SL_Price<df['Open'].iloc[i]:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True            

    exit_len = len(df['Exit'][df['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df['Trade Compeletion Days'][df['Exit'] == True] = df['Exit'][df['Exit'] == True].index - df['Entry'][df['Entry'] == True][:exit_len].index
    return df


# Last Bars base SL Exit Strategy (For Short Signal)
def exit_shortsignal(df=None, entries=None, rr=1, Num_Bars=10):
    
    """
        Calculate exit signals by calculating stoploss from last x num of bars

        Args :
            df (Dataframe): The dataframe of dataset
            entries (Sieres): Entries signals sieres 
            rr (float): risk/reward ratio
            num_bar (int): The last x number of bars from entry signal point

        Return
            df (dataframe): The dataframe having entries & exits signals, days to compelete
                            each trade and status of each trade. 
    """

    df['Entry'] = False
    df['Entry Price'] = df['Close']
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"
    
    # calculate stoploss, take profit & new entry points 
    df['SL Price'] = np.where(entries == True, df['High'].rolling(Num_Bars).max(), 0)
    df['TP Price'] = np.where(df['SL Price']>0, df['Close'] - (df['Close'] - df['SL Price']).abs() * rr, 0)
    df['Entry Price'] = np.where(entries==True, df['Close'], 0)

    # calculate trade status(win/loss) after adding fee 
    # df['Trade Status (Win/Loss)'] = np.where(((df['Exit Price'][df['Exit'] == True] - df['Entry Price'][df['Entry'] == True][:exit_len])-Fee)>=0,'Win',
    #                                     np.where(((df['Exit Price'][df['Exit'] == True] - df['Entry Price'][df['Entry'] == True][:exit_len])-Fee)<0,'Loss',"Nothing"))

    status = True

    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            status = False
            SL_Price=df['SL Price'].iloc[i]
            TP_Price=df['TP Price'].iloc[i]

            df['Entry'].iloc[i] = True

        elif status == False:
            
            if ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):

                if (df['Open'].iloc[i] < df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = SL_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True
                elif (df['Open'].iloc[i] > df['Close'].iloc[i]):
                    df['Exit Price'].iloc[i] = TP_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True
            
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = TP_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                status=True

            elif ((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                  ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                df['Exit Price'].iloc[i] = SL_Price
                df['Exit'].iloc[i] = True
                df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                status=True

             
            elif ~((SL_Price >= df['Low'].iloc[i]) and (SL_Price <= df['High'].iloc[i])) and \
                    ~((TP_Price >= df['Low'].iloc[i]) and (TP_Price <= df['High'].iloc[i])):
                if df['Open'].iloc[i]<TP_Price:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True

                elif SL_Price<df['Open'].iloc[i]:
                    df['Exit Price'].iloc[i] = df['Open'].iloc[i]
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Loss"
                    status=True            

    exit_len = len(df['Exit'][df['Exit'] == True])      # total exit signals
    
    # Calculate trading days for each trade 
    df['Trade Compeletion Days'][df['Exit'] == True] = df['Exit'][df['Exit'] == True].index - df['Entry'][df['Entry'] == True][:exit_len].index
    return df


# Function to calculate exit signals on the number of bars (For Short Signal)
# -----------------------------  Start  -----------------------------
def bar_base_shortexit(df = None, entries = None, num_bars = None):
    
    """
        calculate exit signals with x num of bars 

        Args :
            price (Series): The Series of closing or other price
            entries (Sieres): Entries signals sieres
            num_bars (float): num of bars

        Return
            new_entries (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            exit_prices (Series float64): exit signals prices
    """
    df['Entry'] = False
    df['Entry Price'] = 0
    df['Exit'] = False
    df['Exit Price'] = 0
    df['Trade Compeletion Days'] = 0
    df['Trade Status (Win/Loss)'] = "Nothing"

    status=True
    count_bars=0
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            count_bars=1
            status=False
            df['Entry'].iloc[i] = True

        
        elif status==False and count_bars==num_bars:
            status=True
            df['Exit Price'].iloc[i] = df['Close'].iloc[i]
            df['Exit'].iloc[i] = True
            df['Trade Status (Win/Loss)'].iloc[i] = np.where((df['Exit Price'].iloc[i]-df['Entry Price'].iloc[i-num_bars])>0,"Loss","Win")
            count_bars=1
            
        else:
            count_bars+=1

    df['Entry Price'] = np.where((df['Entry']==True),df['Close'],0)
    return df

# -----------------------------  End  -----------------------------