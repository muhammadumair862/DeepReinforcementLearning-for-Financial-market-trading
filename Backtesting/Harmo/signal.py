import pandas as pd
from scipy import stats
import numpy as np
from datetime import timedelta
import vectorbt as vbt
import talib
from datetime import datetime, time

# Function to combine entries generate by 2 different logics/indicators crossover
# -----------------------------  Start  -----------------------------

def combine_entries(entries1=None, entries2=None):

    """
        This function will combine entries generate by 2 different logics/indicators crossover

        Args :
            entries1 (series): entries of first logic
            entries2 (series): entries of second logic
        Return :
            entries (series): series of entries
    """

    fil_entries,indexes=[],[]
    for i,j in zip(entries1,entries2):
        # print(i)
        if i==True and j==True:
            fil_entries.append(i)
            # indexes.append(ind)
        else:
            fil_entries.append(False)
    entries=pd.Series(fil_entries,index=entries1.index)
    entries.index.name='Open time'
    return entries

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



# Function to get Ohlc and avg hlc values
# -----------------------------  Start  -----------------------------

def ohlc_values(data=None):

    """
        This function get ohlc prices from data and calculate avg of ohl prices

        Args :
            data (Dataframe): Dataframe of data
        
        Return :
            open_price (Series): series of opening price of bar/candle
            high_price (Series): series of closing price of bar/candle
            low_price (Series): series of low price of bar/candle
            close_price (Series): series of closing price of bar/candle
            avg_price (Series): series of average price of open, high, low price
    """

    open_price = data.get('Open')
    high_price = data.get('High')
    low_price = data.get('Low')
    close_price = data.get('Close')
    avg_price=(high_price+low_price+close_price)/3
    return open_price, high_price, low_price, close_price, avg_price

# -----------------------------  End  -----------------------------



# Function to calculate moving linear regression
# -----------------------------  Start  -----------------------------

def LinearRegression_Calculate(x,slope,intercept):
    return slope * x + intercept

def verify(y,period=14):
    x = list(range(period))
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    return LinearRegression_Calculate(period,slope=slope,intercept=intercept)

def LinReg(price=None,period=14):
    
    """
        function to calculate linear regression

        Args :
            price (Series): bydefault closing price but it can be open/high/low
            period (int): number period to calculate linear regression

        Return :
            linreg (Series): predict series using linear regression
    """

    linreg = price.rolling(period).apply(verify, args=(period,))
    return linreg

# -----------------------------  End  -----------------------------



# Function to Calculate Stoploss & Take Profit Price
# -----------------------------  Start  -----------------------------

def Stoploss(df=None,counter=None,SL_range=None,column='Low'):
    SL_Price=min(df[column].iloc[counter-SL_range:counter])
    return SL_Price

def Take_Profit(df=None,counter=None,SL_Price=None,column='Close',RR=1):
    SL=abs(df[column].iloc[counter]-SL_Price)
    TP=RR*SL         # Risk Reward Ratio
    TP_Price=TP+df[column].iloc[counter]
    return TP_Price

# -----------------------------  End  -----------------------------



# Function to Sell on custom logic
# -----------------------------  Start  -----------------------------

def BarRange(df,counter,value):
    """
        This Function is use to check the value inside current bar or not
        Input : data, index location, current price
        output: True/False     
    """
    
    if df['High'].iloc[counter] >= value >=df['Low'].iloc[counter] :
        return True
    return False

def TP_point(df,counter,TP):
    
    if df.Open.iloc[counter]<df.High.iloc[counter-1] and df.Open.iloc[counter]<=TP:
        return True
    else:
        return False

def SL_point(df,counter,SL):
    if df.Open.iloc[counter]>df.Low.iloc[counter-1] and df.Open.iloc[counter]>=SL:
        return True
    else:
        return False

def uptrend(df,counter):
    if df['Open'].iloc[counter]>df['Open'].iloc[counter-1]:
        return True
    else:
        return False

def downtrend(df,counter):
    if df['Open'].iloc[counter]<df['Open'].iloc[counter-1]:
            return True
    else:
            return False

# def exit_signal(df=None,entries=None,entry_column='Close',RR=1,last_bars=10):

#     prices=np.array([])
#     entries1=np.array([],dtype=bool)
#     exits=np.array([],dtype=bool)
#     status=True
#     SL_Price,TP_Price=0,0

#     #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
#     for i in range(len(entries)):
#         # Find Exit Prices of Enteries Signals
#         if entries.iloc[i] and status:
#             SL_Price=Stoploss(df=df,counter=i,SL_range=last_bars)
#             TP_Price=Take_Profit(df=df,counter=i,SL_Price=SL_Price,RR=RR)
#             # print('Original Price :',df['Close'].iloc[i])
#             # print('Stoploss :',SL_Price)
#             # print('Take Profit :',TP_Price)
#             status=False

#             entries1=np.append(entries1,True)
#             exits=np.append(exits,False)
#             prices=np.append(prices,0)
    
#             # break
        
#         elif status==False:
#             entries1=np.append(entries1,False)
            
#             if BarRange(df,i,TP_Price) and BarRange(df,i,SL_Price):
#                 if uptrend(df,i):
#                     prices=np.append(prices,TP_Price)
#                     exits=np.append(exits,True)
#                     status=True
#                     # print("Win1")
#                 elif downtrend(df,i):
#                     prices=np.append(prices,SL_Price)
#                     exits=np.append(exits,True)
#                     status=True
#                     # print("Loss1")
#                 else:
#                     exits=np.append(exits,False)
#                     prices=np.append(prices,0)
            
#             elif BarRange(df,i,TP_Price) and not BarRange(df,i,SL_Price):
#                 prices=np.append(prices,TP_Price)
#                 exits=np.append(exits,True)
#                 status=True
#                 # print("Win")

#             elif BarRange(df,i,SL_Price) and not BarRange(df,i,TP_Price):
#                 prices=np.append(prices,SL_Price)
#                 exits=np.append(exits,True)
#                 status=True

             
#             elif not BarRange(df,i,TP_Price) and not BarRange(df,i,SL_Price):
#                 if df['Open'].iloc[i]>TP_Price:
#                     exits=np.append(exits,True)
#                     prices=np.append(prices,df['Open'].iloc[i])
#                     status=True

#                 elif SL_Price>df['Open'].iloc[i]:
#                     prices=np.append(prices,df['Open'].iloc[i])
#                     exits=np.append(exits,True)
#                     status=True
#                 else:
#                     exits=np.append(exits,False)
#                     prices=np.append(prices,0)
        
#             else:
#                 exits=np.append(exits,False)
#                 prices=np.append(prices,0)
#         else:
#             entries1=np.append(entries1,False)
#             exits=np.append(exits,False)
#             prices=np.append(prices,0)
            
      

#     exits=pd.Series(exits,index=df.index)
#     prices=pd.Series(prices,index=df.index)
#     entries1=pd.Series(entries1,index=df.index)
#     entry_prices=df['Close']
#     return entries1,exits,entry_prices,prices
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

                if (df['Low'].iloc[i] < df['High'].iloc[i]):
                    df['Exit Price'].iloc[i] = TP_Price
                    df['Exit'].iloc[i] = True
                    df['Trade Status (Win/Loss)'].iloc[i] = "Win"
                    status=True
                elif (df['Low'].iloc[i] > df['High'].iloc[i]):
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


# -----------------------------  End  -----------------------------



# Function use for conversion of dataframe into different dataframes
# -----------------------------  Start  -----------------------------

def timeframe(df = None, timeframe = None, year = None, 
                         month = None, week = None, day = None, 
                         hour = None, min = None, sec = None):
    
    """
        Convert the pandas dataframe timeframe into another timeframe (smaller -> higher)

        Args :
            df (Dataframe): The dataframe of dataset
            timeframe (str): timeframe like '1h', '1d' etc.. 
            year (int): num to convert timefarme into yearly timeframe
            month (int): num to convert timefarme into yearly timeframe
            week (int): num to convert timefarme into yearly timeframe
            day (int): num to convert timefarme into yearly timeframe
            hour (int): num to convert timefarme into yearly timeframe
            min (int): num to convert timefarme into yearly timeframe
            sec (int): num to convert timefarme into second timeframe

        Return
            tf_df (dataframe): dataframe with newly timeframe 
    """

    if timeframe is None:
        if year:
            timeframe = str(year) + 'y'
        elif month:
            timeframe = str(month) + 'M'
        elif week:
            timeframe = str(week) + 'W'
        elif day:
            timeframe = str(day) + 'd'
        elif hour:
            timeframe = str(hour) + 'h'
        elif min:
            timeframe = str(min) + 'min'
        elif sec:
            timeframe = str(sec) + 's'
        else:
            return df

    try:
        tf_df = df.resample(timeframe).agg({
            "Open" : "first",
            "High" : "max",
            "Low" : "min",
            "Close" : "last",
            "Volume" : "sum"
        })

    except:
        print ("Incorrect Time Frame. Kindly use correct timeframe!!!")
        return None

    tf_df = tf_df.dropna()
    return tf_df

# -----------------------------  End  -----------------------------



# Function to calculate max win Strikes in row
# -----------------------------  Start  -----------------------------

def max_win_row(price = None, entries = None, exits = None):
    
    """
        Calculate consecutive wins trades

        Args :
            price (Dataframe): The dataframe contain entries & exits
            entries (Sieres): Entries signals sieres
            exits (Sieres): Exits signals sieres 

        Return
            win_df (dataframe): The dataframe having consecutive win trades 
    """

    m = pd.DataFrame()
    m['Dif'] = np.array(price[entries])-np.array(price[exits])

    b4t_start, b4t_end, b4t_amount, b4t_avg, win_strikes = [], [], [], [], []
    status = 0
    tw = 0

    for i in range(1, len(m)):
        if m.Dif.iloc[i] > 0:
            tw += 1
            status = 1
        elif m.Dif.iloc[i] < 0 and status == 1 and tw > 1:
            b4t_start.append(m.index[i - 1 - (tw) * 2])
            b4t_end.append(m.index[i-2])
            win_strikes.append(tw)
            sum=0
            for k in range(1,tw+1):
                sum+=m.Dif.iloc[i-k*2]
            b4t_amount.append(abs(sum))
            b4t_avg.append(abs(sum)/tw)
            tw = 0
            status = 0
        else:
            tw = 0

    win_df = pd.DataFrame({'Opening Position':b4t_start,'Closing Position':b4t_end,'No. of Strikes in Row':win_strikes,'Cash Gain':b4t_amount,'Average Cash Gain':b4t_avg})
    return win_df.sort_values(by="No. of Strikes in Row", ascending=False)

# -----------------------------  End  -----------------------------



# Function to calculate max Loss Strikes in row
# -----------------------------  Start  -----------------------------

def max_loss_row(price=None,entries=None,exits=None):
    
    """
        Calculate consecutive loss trades 

        Args :
            price (Dataframe): The dataframe contain entries & exits
            entries (Sieres): Entries signals sieres
            exits (Sieres): Exits signals sieres 

        Return
            loss_df (dataframe): The dataframe having consecutive loss trades 
    """
    
    loss = pd.DataFrame()
    loss['Dif'] = np.array(price[entries])-np.array(price[exits])
    b4l_start,b4l_end,b4l_amount,b4l_avg,loss_strikes=[],[],[],[],[]
    status=0
    tl=0
    for i in range(1,len(loss)):
        if loss.Dif.iloc[i] < 0:
            tl+=1
            status=1
        elif loss.Dif.iloc[i] > 0 and status == 1 and tl > 1:
            b4l_start.append(loss.index[i - 1 - (tl) * 2])
            b4l_end.append(loss.index[i - 2])
            loss_strikes.append(tl)
            sum = 0
            for k in range(1, tl + 1):
                sum += loss.Dif.iloc[i - k * 2]
            b4l_amount.append(abs(sum))
            b4l_avg.append(abs(sum) / tl)
            tl = 0
            status = 0
        else:
            tl = 0

    loss_df = pd.DataFrame({'Opening Position':b4l_start, 'Closing Position':b4l_end,
                            'No. of Strikes in Row':loss_strikes,'Cash loss':b4l_amount,
                            'Average Cash loss':b4l_avg})
    return loss_df.sort_values(by="No. of Strikes in Row", ascending=False)

# -----------------------------  End  -----------------------------



# Function to calculate max Loss Strikes in row
# -----------------------------  Start  -----------------------------

def winloss(x):
    """
    function to calculate trade status
    """
    if x>0:
        return 'Win'
    elif x==0:
        return 'Neutral'
    else:
        return 'Loss'

def signals_files(entry_prices = None, exit_prices = None, entries = None,
                    exits = None, fee = None, filename = 'output.csv'):
    
    """
        Calculate trade status, profit/loss amount & save in csv file

        Args :
            entry_price (Series float64): The series contain entry prices
            entry_price (Series float64): The series contain exit prices
            entries (Sieres bool): Entries signals sieres
            exits (Sieres bool): Exits signals sieres
            fee (int): each trade transcation fee
            filename (str): name of csv file 

        Return
            df (dataframe): The dataframe having trades profit/loss amount with trade status  
    """

    en1 = entry_prices[entries.index[entries == True]]
    ex1 = exit_prices[exits.index[exits == True]]

    df = pd.DataFrame({'Entry Time':en1.index, 'Exit Time':ex1.index, 'Entry Signal Price':en1.to_list(),
                        'Exit Signal Price':ex1.to_list()})
    df['Profit/Loss Amount'] = df['Exit Signal Price'] - df['Entry Signal Price']
    df['Fee Amount'] = fee
    df['Total Profit/Loss Amount'] = df['Profit/Loss Amount']-fee
    df['Win/Loss']= df['Total Profit/Loss Amount'].apply(winloss)
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    df.to_csv(filename,index=False)
    return df

# -----------------------------  End  -----------------------------



# Function to calculate exit signals on the basis of time
# -----------------------------  Start  -----------------------------
def time_base_exit(price = None, entries = None, hour = None,
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

    entry_prices=np.array([])
    exit_prices=np.array([])
    new_entries=np.array([],dtype=bool)
    exits=np.array([],dtype=bool)
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

            new_entries=np.append(new_entries,True)
            exits=np.append(exits,False)
            entry_prices=np.append(entry_prices,price.loc[i])
            exit_prices=np.append(exit_prices,0)
            
            status=False
        
        elif status==False and Ex_time<=i:
            # print("Sell",i)

            Ex_time=0
            status=True
            new_entries=np.append(new_entries,False)
            exits=np.append(exits,True)
            entry_prices=np.append(entry_prices,0)
            exit_prices=np.append(exit_prices,price.loc[i])
        else:
            new_entries=np.append(new_entries,False)
            exits=np.append(exits,False)
            entry_prices=np.append(entry_prices,0)
            exit_prices=np.append(exit_prices,0)
    
    new_entries=pd.Series(new_entries,index=price.index)
    exits=pd.Series(exits,index=price.index)
    entry_prices=pd.Series(entry_prices,index=price.index)
    exit_prices=pd.Series(exit_prices,index=price.index)
    return new_entries,exits,entry_prices,exit_prices

# -----------------------------  End  -----------------------------



# Function to calculate exit signals on the number of bars
# -----------------------------  Start  -----------------------------
def bar_base_exit(price = None, entries = None, num_bars = None):
    
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

    entry_prices=np.array([])
    exit_prices=np.array([])
    new_entries=np.array([],dtype=bool)
    exits=np.array([],dtype=bool)
    status=True
    count_bars=0
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in entries.index:
        # Find Exit Prices of Enteries Signals
        if entries.loc[i] and status:
            new_entries=np.append(new_entries,True)
            exits=np.append(exits,False)
            entry_prices=np.append(entry_prices,price.loc[i])
            exit_prices=np.append(exit_prices,0)
            count_bars=0
            
            status=False
        
        elif status==False and count_bars==num_bars:
            # print("Sell",i)
            count_bars=0
            status=True
            new_entries=np.append(new_entries,False)
            exits=np.append(exits,True)
            entry_prices=np.append(entry_prices,0)
            exit_prices=np.append(exit_prices,price.loc[i])
        else:
            new_entries=np.append(new_entries,False)
            exits=np.append(exits,False)
            entry_prices=np.append(entry_prices,0)
            exit_prices=np.append(exit_prices,0)
            count_bars+=1
    
    new_entries=pd.Series(new_entries,index=price.index)
    exits=pd.Series(exits,index=price.index)
    entry_prices=pd.Series(entry_prices,index=price.index)
    exit_prices=pd.Series(exit_prices,index=price.index)
    return new_entries,exits,entry_prices,exit_prices

# -----------------------------  End  -----------------------------




# Function to Sell on with fixed stoploss
# -----------------------------  Start  -----------------------------

def exit_with_SL(df=None, entries=None, entry_column='Close', 
                    RR=1, SL=None):
    """
        calculate exit signals with stoploss that calculate from zigzag (contain highs & lows) 

        Args :
            df (Dataframe): The dataframe of data
            entries (Sieres): Entries signals sieres
            RR (float): risk reward ratio
            SL (float): Stoploss amount

        Return
            entries1 (Series bool): New entry signal 
            exits (Series bool): exit signals
            prices (Series float64): exit signals prices
    """
    prices=np.array([])
    entries1=np.array([],dtype=bool)
    exits=np.array([],dtype=bool)
    status=True
    SL_Price, TP_Price=0,0
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            SL_Price=SL+df[entry_column].iloc[i]
            TP_Price=Take_Profit(df=df,counter=i,SL_Price=SL_Price,RR=RR)
            # print('Original Price :',df['Close'].iloc[i])
            # print('Stoploss :',SL_Price)
            # print('Take Profit :',TP_Price)
            status=False

            entries1=np.append(entries1,True)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
            # break
        
        elif status==False:
            entries1=np.append(entries1,False)

            
            if BarRange(df,i,TP_Price) and BarRange(df,i,SL_Price):
                if uptrend(df,i):
                    prices=np.append(prices,TP_Price)
                    exits=np.append(exits,True)
                    status=True
                    # print("Win1")
                elif downtrend(df,i):
                    prices=np.append(prices,SL_Price)
                    exits=np.append(exits,True)
                    status=True
                    # print("Loss1")
                else:
                    exits=np.append(exits,False)
                    prices=np.append(prices,0)
            
            elif BarRange(df,i,TP_Price) and not BarRange(df,i,SL_Price):
                prices=np.append(prices,TP_Price)
                exits=np.append(exits,True)
                status=True
                # print("Win")

            elif BarRange(df,i,SL_Price) and not BarRange(df,i,TP_Price):
                prices=np.append(prices,SL_Price)
                exits=np.append(exits,True)
                status=True
                # print("Loss")

            # elif BarRange(df,i,TP_Price) and BarRange(df,i,SL_Price):
            #     if uptrend(df,i):
            #         prices=np.append(prices,TP_Price)
            #         exits=np.append(exits,True)
            #         status=True
            #         # print("Win1")
            #     elif downtrend(df,i):
            #         prices=np.append(prices,SL_Price)
            #         exits=np.append(exits,True)
            #         status=True
            #         # print("Loss1")
            #     else:
            #         if df['Close'].iloc[i]>df['Open'].iloc[i]:
            #             prices=np.append(prices,TP_Price)
            #             exits=np.append(exits,True)
            #             status=True
            #             # print("Win11")
            #         else:
            #             prices=np.append(prices,SL_Price)
            #             exits=np.append(exits,True)
            #             status=True
            #             # print("Loss11")
             
            elif not BarRange(df,i,TP_Price) and not BarRange(df,i,SL_Price):
                if df['Open'].iloc[i]>TP_Price:
                    exits=np.append(exits,True)
                    prices=np.append(prices,df['Open'].iloc[i])

                elif SL_Price>df['Open'].iloc[i]:
                    prices=np.append(prices,df['Open'].iloc[i])
                    exits=np.append(exits,True)
                else:
                    exits=np.append(exits,False)
                    prices=np.append(prices,0)




                
            #     status=True
            #     # print("loss2",i)
        
            else:
                exits=np.append(exits,False)
                prices=np.append(prices,0)
        else:
            entries1=np.append(entries1,False)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
      

    exits=pd.Series(exits,index=df.index)
    prices=pd.Series(prices,index=df.index)
    entries1=pd.Series(entries1,index=df.index)

    # d1=pd.DataFrame({''})
    return entries1 , exits , prices

# -----------------------------  End  -----------------------------



# Function to to give highs & Lows of zigzag
# -----------------------------  Start  -----------------------------
def zigzag_points(df_high, df_low):
    
    """
        calculate highs & lows 

        Args :
            df_high (Series): High prices
            df_low (Sieres): Low prices

        Return
            highs (Series): Highs of data 
            lows (Series): Lows of data
    """
    
    zigzag_points=vbt.PIVOTINFO.run(df_high, df_low, 0.01, 0.01).conf_value.drop_duplicates().dropna()
    if zigzag_points.iloc[0]<zigzag_points.iloc[1]:
        highs=zigzag_points[1::2]
        lows=zigzag_points[::2]
    else:
        highs=zigzag_points[::2]
        lows=zigzag_points[1::2]

    return highs, lows

# -----------------------------  End  -----------------------------



# Function to to give highs & Lows of zigzag
# -----------------------------  Start  -----------------------------
def exit_zigzag(df = None, entries = None, RR = None):
    """
        calculate exit signals with stoploss that calculate from zigzag (contain highs & lows) 

        Args :
            df (Dataframe): The dataframe of data
            entries (Sieres): Entries signals sieres
            RR (float): risk reward ratio

        Return
            entries1 (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            prices (Series float64): exit signals prices
    """

    prices = np.array([])
    entries1 = np.array([],dtype=bool)
    exits = np.array([],dtype=bool)
    status = True
    SL_Price, TP_Price = 0, 0
    start_time='11 AM'
    end_time='4 PM'

    highs, lows = zigzag_points(df['High'],df['Low'])
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            SL_Price=max(lows[lows.index<df.index[i]])
            # SL_Price=Stoploss(df=df,counter=i,SL_range=)
            TP_Price=Take_Profit(df=df,counter=i,SL_Price=SL_Price,RR=RR)
            # print('Original Price :',df['Close'].iloc[i])
            # print('Stoploss :',SL_Price)
            # print('Take Profit :',TP_Price)
            status=False

            entries1=np.append(entries1,True)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
        elif status==False:
            entries1=np.append(entries1,False)
            if df.Low.iloc[i]<=SL_Price and df.Open.iloc[i]>=SL_Price:
                prices=np.append(prices,TP_Price)
                exits=np.append(exits,True)
                status=True
            elif df.High.iloc[i]>=TP_Price and df.Open.iloc[i]<=TP_Price:
                prices=np.append(prices,TP_Price)
                exits=np.append(exits,True)
                status=True
            else:
                prices=np.append(prices,df.Open.iloc[i])
                exits=np.append(exits,False)
        else:
            entries1=np.append(entries1,False)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
    
    exits=pd.Series(exits,index=df.index)
    prices=pd.Series(prices,index=df.index)
    entries1=pd.Series(entries1,index=df.index)
    entry_prices=df['Close'][entries==True]
    # d1=pd.DataFrame({''})
    return entries1,exits,entry_prices,prices

# -----------------------------  End  -----------------------------



# Function to to give highs & Lows of zigzag (with time constraint)
# -----------------------------  Start  -----------------------------
def exit_time_zigzag(df=None,entries=None,RR=None,start_time='00:00',end_time='23:59'):
    
    """
        calculate exit signals with stoploss that calculate from zigzag (contain highs & lows) 
        with time constraint

        Args :
            df (Dataframe): The dataframe of data
            entries (Sieres): Entries signals sieres
            RR (float): risk reward ratio
            start_time (str): starting time
            end_time (str): ending time

        Return
            entries1 (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            prices (Series float64): exit signals prices
    """

    prices = np.array([])
    entries1 = np.array([],dtype=bool)
    exits = np.array([],dtype=bool)
    status = True
    SL_Price, TP_Price = 0, 0
    start_time = datetime.strptime(start_time, "%H:%M")
    end_time = datetime.strptime(end_time, "%H:%M")

    start_time = time(start_time.hour, start_time.minute)
    end_time = time(end_time.hour, end_time.minute)
    if end_time.minute+15>=60:
        z=end_time.minute+15-60
        if end_time.hour+1>=24:
            ext_time = time(0, z)
        else:
            ext_time = time(end_time.hour+1, z)
    else:
        ext_time = time(end_time.hour, end_time.minute+15)

    highs, lows = zigzag_points(df['High'],df['Low'])
    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status and (start_time<=entries.index[i].time() and end_time>=entries.index[i].time()):
            # print(entries.index[i])
            SL_Price=max(lows[lows.index<df.index[i]])
            TP_Price=Take_Profit(df=df,counter=i,SL_Price=SL_Price,RR=RR)
            status=False

            entries1=np.append(entries1,True)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
        elif status==False and (start_time<=entries.index[i].time() and ext_time>=entries.index[i].time()):
            entries1=np.append(entries1,False)
            if df.Low.iloc[i]<=SL_Price and df.Open.iloc[i]>=SL_Price:
                prices=np.append(prices,TP_Price)
                exits=np.append(exits,True)
                status=True
            elif df.High.iloc[i]>=TP_Price and df.Open.iloc[i]<=TP_Price:
                prices=np.append(prices,TP_Price)
                exits=np.append(exits,True)
                status=True
            else:
                prices=np.append(prices,df.Open.iloc[i])
                exits=np.append(exits,False)
        elif status==False and (ext_time<entries.index[i].time()):
            entries1=np.append(entries1,False)
            exits=np.append(exits,True)
            prices=np.append(prices,df.Close.iloc[i])
            status=True
        else:
            entries1=np.append(entries1,False)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
    
    exits=pd.Series(exits,index=df.index)
    prices=pd.Series(prices,index=df.index)
    entries1=pd.Series(entries1,index=df.index)
    entry_prices=df['Close'][entries==True]

    return entries1,exits,entry_prices,prices

# -----------------------------  End  -----------------------------






# Function to calculate exit signals on the basis special rsi indicator indicator
# -----------------------------  Start  -----------------------------
def indicator_base_exit(price=None,entries=None,linreg_period=160,ma_period=120,
                            rsi_period=150,trade='long'):
    
    """
    price               : input price like open, high, low & close price etc
    ma_period           : ma period to calculate ma
    linreg_period       : period to calculate moving linear regression
    rsi_period          : to calculate the rsi value
    trade       : trade to identify poaition is long or short

    """

    entry_prices=np.array([])
    exit_prices=np.array([])
    new_entries=np.array([],dtype=bool)
    exits=np.array([],dtype=bool)
    # Defining INDICATORS to be use
    # Calculate RSI crossover with MA of RSI
    vbt.talib('RSI')
    rsi=vbt.RSI.run(price,window=rsi_period)
    vbt.talib('MA')
    rsi_ma=vbt.MA.run(rsi.rsi,window=ma_period)
    # Calculate Price above moving LinReg 
    linreg=LinReg(price,period=linreg_period)

    # Defining Rules
    if trade=='short':
        en1=rsi.rsi.vbt.crossed_above(rsi_ma.ma)
        en2=price > linreg
    elif trade=='long':
        en1=rsi.rsi.vbt.crossed_below(rsi_ma.ma)
        en2=price < linreg
    

    # Defining Entries logic
    exits=combine_entries(en1,en2)

    new_entries=entries
    entry_prices=price[entries==True]
    exit_prices=price[exits==True]
    return new_entries,exits,entry_prices,exit_prices



# def macd_base_exit(price=None,entries=None,fast=60,slow=120,
#                         ma_period=150,macd_period=150,trade='long'):
    
#     """
#     price               : input price like open, high, low & close price etc
#     ma_period           : ma period to calculate ma
#     linreg_period       : period to calculate moving linear regression
#     rsi_period          : to calculate the rsi value
#     trade       : trade to identify poaition is long or short

#     """
    
#     entry_prices=np.array([])
#     exit_prices=np.array([])
#     new_entries=np.array([],dtype=bool)
#     exits=np.array([],dtype=bool)
#     # Defining INDICATORS to be use

#     WMA=vbt.talib("WMA").run(price,ma_period).real.to_numpy()
#     SMA=vbt.MA.run(price,ma_period)
#     Special_LinReg=3*WMA-2*SMA.ma

#     WMAF=vbt.talib("WMA").run(price,fast).real.to_numpy()
#     SMAF=vbt.MA.run(price,fast)
#     Special_LinRegF=3*WMAF-2*SMAF.ma

#     WMAS=vbt.talib("WMA").run(price,ma_period).real.to_numpy()
#     SMAS=vbt.MA.run(price,ma_period)
#     Special_LinRegS=3*WMAS-2*SMAS.ma

#     MACD = Special_LinRegF - Special_LinRegS
#     MACD_signal=vbt.MA.run(MACD,macd_period)

#     # Defining Rules
#     if trade=='short':
#         en1=MACD.vbt.crossed_above(MACD_signal.ma)
#         en2=price > Special_LinReg
#     elif trade=='long':
#         en1=MACD.vbt.crossed_above(MACD_signal.ma)
#         en2=price > Special_LinReg

#     # Defining Entries logic
#     exits=combine_entries(en1,en2)

#     new_entries=entries
#     entry_prices=price[entries==True]
#     exit_prices=price[exits==True]
#     return new_entries,exits,entry_prices,exit_prices

def macd_base_exit(price = None, entries = None, fast=12, 
                    slow = 26, signal_period = 9, trade = 'long'):
    
    """
        calculate exit signals on the basis of macd indicator 

        Args :
            price (Series): The Series of closing or other prices
            entries (Sieres): Entries signals sieres
            fast (int): fast period for macd
            slow (int): slow period for macd
            signal_period (int): period for macd signal


        Return
            new_entries (Series bool): New entry signal 
            exits (Series bool): exit signals
            entry_prices (Series float64): entry signal prices
            exit_prices (Series float64): exit signals prices
    """    

    macd, macdsignal, macdhist = talib.MACDEXT(price, fastperiod = fast, fastmatype = 0, 
                                                slowperiod = slow, slowmatype = 0, 
                                                signalperiod = signal_period, signalmatype=0)
    if trade=='short':
        exits=macd.vbt.crossed_above(macdsignal)
    elif trade=='long':
        exits=macd.vbt.crossed_below(macdsignal)

    new_entries=entries
    entry_prices=price[entries==True]
    exit_prices=price[exits==True]
    return new_entries, exits, entry_prices, exit_prices



# def exit_signalSLRR(df=None,entries=None,RR=None ,last_bars=10):
#     prices=np.array([])
#     entries1=np.array([],dtype=bool)
#     exits=np.array([],dtype=bool)
#     status=True
#     SL_Price,TP_Price=0,0
#     #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
#     for i in range(len(entries)):
#         # Find Exit Prices of Enteries Signals
#         if entries.iloc[i] and status:
#             SL_Price=Stoploss(df=df,counter=i,SL_range=last_bars)
#             TP_Price=Take_Profit(df=df,counter=i,SL_Price=SL_Price,RR=RR)
#             # print('Original Price :',df['Close'].iloc[i])
#             # print('Stoploss :',SL_Price)
#             # print('Take Profit :',TP_Price)
#             status=False

#             entries1=np.append(entries1,True)
#             exits=np.append(exits,False)
#             prices=np.append(prices,0)
#         elif status==False:
#             entries1=np.append(entries1,False)
#             if df.Low.iloc[i]<=SL_Price and df.Open.iloc[i]>=SL_Price:
#                 prices=np.append(prices,TP_Price)
#                 exits=np.append(exits,True)
#                 status=True
#             elif df.High.iloc[i]>=TP_Price and df.Open.iloc[i]<=TP_Price:
#                 prices=np.append(prices,TP_Price)
#                 exits=np.append(exits,True)
#                 status=True
#             else:
#                 prices=np.append(prices,df.Open.iloc[i])
#                 exits=np.append(exits,False)
#         else:
#             entries1=np.append(entries1,False)
#             exits=np.append(exits,False)
#             prices=np.append(prices,0)
    
#     exits=pd.Series(exits,index=df.index)
#     prices=pd.Series(prices,index=df.index)
#     entries1=pd.Series(entries1,index=df.index)
#     entry_prices=df['Close'][entries==True]
#     # d1=pd.DataFrame({''})
#     return entries1,exits,entry_prices,prices





    #  $$$$$$$$$  Fixed amount with rr

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

    prices=np.array([])
    entries1=np.array([],dtype=bool)
    exits=np.array([],dtype=bool)
    status=True
    SL_Price,TP_Price=0,0

    #   $$$$$$$$$$$$  Logic Part  $$$$$$$$$$$$$$
    
    for i in range(len(entries)):
        # Find Exit Prices of Enteries Signals
        if entries.iloc[i] and status:
            # SL_Price=Stoploss(df=df,counter=i,SL_range=last_bars)
            SL_Price=df['Close'].iloc[i]-SL
            TP_Price=Take_Profit(df=df,counter=i,SL_Price=SL_Price,RR=RR)
            status=False

            entries1=np.append(entries1,True)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
            # break
        
        elif status==False:            
            entries1=np.append(entries1,False)

            
            if BarRange(df,i,TP_Price) and BarRange(df,i,SL_Price):
                if uptrend(df,i):
                    prices=np.append(prices,TP_Price)
                    exits=np.append(exits,True)
                    status=True
                    # print("Win1")
                elif downtrend(df,i):
                    prices=np.append(prices,SL_Price)
                    exits=np.append(exits,True)
                    status=True
                    # print("Loss1")
                else:
                    exits=np.append(exits,False)
                    prices=np.append(prices,0)
            
            elif BarRange(df,i,TP_Price) and not BarRange(df,i,SL_Price):
                prices=np.append(prices,TP_Price)
                exits=np.append(exits,True)
                status=True
                # print("Win")

            elif BarRange(df,i,SL_Price) and not BarRange(df,i,TP_Price):
                prices=np.append(prices,SL_Price)
                exits=np.append(exits,True)
                status=True
                # print("Loss")
             
            elif not BarRange(df,i,TP_Price) and not BarRange(df,i,SL_Price):
                if df['Open'].iloc[i]>TP_Price:
                    exits=np.append(exits,True)
                    prices=np.append(prices,df['Open'].iloc[i])
                    status=True

                elif SL_Price>df['Open'].iloc[i]:
                    prices=np.append(prices,df['Open'].iloc[i])
                    exits=np.append(exits,True)
                    status=True
                else:
                    exits=np.append(exits,False)
                    prices=np.append(prices,0)
                    

            else:
                exits=np.append(exits,False)
                prices=np.append(prices,0)
        else:
            entries1=np.append(entries1,False)
            exits=np.append(exits,False)
            prices=np.append(prices,0)
      

    exits=pd.Series(exits,index=df.index)
    prices=pd.Series(prices,index=df.index)
    entries1=pd.Series(entries1,index=df.index)
    entry_prices=df['Close']
    # d1=pd.DataFrame({''})
    return entries1,exits,entry_prices,prices


# @@@@@@@@@@  Enteries Signal Files Generation  @@@@@@@@@
def entries_files(df, entries, exits, entry_prices, exit_prices, filename="signals.csv"):
    
    """
        Store entry, exit signals with their prices & target in csv file.

        Args :
            entry_price (Series float64): The series contain entry prices
            entry_price (Series float64): The series contain exit prices
            entries (Sieres bool): Entries signals sieres
            exits (Sieres bool): Exits signals sieres
            filename (str): name of csv file  

        output
            create csv having entry & exit signals and their prices. Also target column
    """

    df['entries'] = entries
    df['exits'] = exits
    df['entry_prices'] = entry_prices
    df['exit_prices'] = exit_prices
    df['entry_prices'].fillna(0,inplace=True)
    
    try:
        arr1 = df['entry_prices'][df['entries']==True].to_numpy() - df['exit_prices'][df['exits']==True].to_numpy()
    except:
        arr1 = df['entry_prices'][df['entries']==True].to_numpy()[:-1] - df['exit_prices'][df['exits']==True].to_numpy()

    arr2 = np.where(arr1>0,-1,0) + np.where(arr1<0,1,0)
    ser_aar2=pd.Series(arr2,index=df['entries'][df['entries']==True][:-1].index)
    df['target']=0
    df['target'][ser_aar2.index]=arr2

    df.to_csv(filename)




def create_entries(entries, exits, entry_prices, exit_prices, filename="signals.csv"):
        
    """
        Store entry & exit signals with their prices in csv file.

        Args :
            entry_price (Series float64): The series contain entry prices
            entry_price (Series float64): The series contain exit prices
            entries (Sieres bool): Entries signals sieres
            exits (Sieres bool): Exits signals sieres
            filename (str): name of csv file  

        output
            create csv having entry & exit signals and their prices 
    """

    df=pd.DataFrame({})
    df['entries'] = entries
    df['exits'] = exits
    df['entry_prices'] = entry_prices
    df['exit_prices'] = exit_prices
    # df = df[df['entries']==True]
    df.to_csv(filename)


# def file_generate(data:pd.DataFrame , sr_df:pd.DataFrame):
#     for i in sr_df.index[:2]:
    # print(i)
    # res = ind.run(
    #     close_price=close_price,
    #     fast=i[0],
    #     slow=i[1],
    #     macd_ma_period=i[2], 
    #     ema_period=i[3],
    #     ema_vol_period=i[4],
    #     param_product = True
    #     )
    # entries,exits,entry_prices,exit_prices = signal.exit_signal(data,entries=res.entries,RR=i[5],last_bars=10)
    # filename = "signals_" + "_".join([str(j) for j in i]) + ".csv"
    # entries_files(data, entries, exits, entry_prices, exit_prices, filename)
