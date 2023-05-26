import numpy as np
import pandas as pd
import vectorbt as vbt
import Harmo.signal as signal
import talib
from talib import MA_Type
from numba import njit, jit


def special_linreg_macd(price=None,ma_period=160,fast=50,
                slow=150,macd_period=150,trade='long'):
    
    """
        function to calculate entries point using special linear regression & macd indicator
        
        Args :
            price (Series): input price like open, high, low & close price etc
            ma_period (Series): ma period to calculate wma
            fast (int): fast is fast period to calculate fast linreg
            slow (int): slow is slow period to calculate slow linreg
            macd_period (int): period to calculate macd
            trade (str): trade to identify poaition is long or short
        
        Retrun :
            entries (Series): entries signals
    """

    # Defining INDICATORS to be use
    wma=vbt.talib("WMA").run(price,ma_period).real.to_numpy()
    sma=vbt.MA.run(price,ma_period)
    special_linreg=3*wma-2*sma.ma

    wma_fast=vbt.talib("WMA").run(price,fast).real.to_numpy()
    sma_fast=vbt.MA.run(price,fast)
    special_linreg_fast=3*wma_fast-2*sma_fast.ma

    wma_slow=vbt.talib("WMA").run(price,slow).real.to_numpy()
    sma_slow=vbt.MA.run(price,slow)
    special_linreg_slow=3*wma_slow-2*sma_slow.ma

    macd = special_linreg_fast - special_linreg_slow
    macd_signal=vbt.MA.run(macd,macd_period)

    # # Defining Rules
    if trade=='long':
        en1=macd.vbt.crossed_above(macd_signal.ma)
        en2=price > special_linreg
    elif trade=='short':
        en1=macd.vbt.crossed_below(macd_signal.ma)
        en2=price < special_linreg
        

    # # Defining Entries logic
    entries=signal.combine_entries(en1,en2)
    return entries


def special_linreg_rsi(price=None,special_ma_period=160,ma_period=120,
                rsi_period=150,trade='long'):
    
    """
        function to calculate entries point using special linear regression & rsi indicator
        
        Args :
            price (Series): input price like open, high, low & close price etc
            ma_period (Series): ma period to calculate wma
            special_ma_period (Series): period to calculate moving linear regression (using 3wma-2sma)
            rsi_period (Series): to calculate the rsi value
            trade (str): trade to identify poaition is long or short
        
        Retrun :
            entries (Series): entries signals
    """


    # Defining INDICATORS to be use
    wma=vbt.talib("WMA").run(price,special_ma_period).real.to_numpy()
    sma=vbt.MA.run(price,special_ma_period)
    special_linreg=3*wma-2*sma.ma

    vbt.talib('RSI')
    rsi=vbt.RSI.run(price,window=rsi_period)
    vbt.talib('MA')
    rsi_ma=vbt.MA.run(rsi.rsi,window=ma_period)

    # Defining Rules
    if trade=='long':
        en1=rsi.rsi.vbt.crossed_above(rsi_ma.ma)
        en2=price > special_linreg
    elif trade=='short':
        en1=rsi.rsi.vbt.crossed_below(rsi_ma.ma)
        en2=price < special_linreg
        
    # Defining Entries logic
    entries=signal.combine_entries(en1,en2)
    return entries

def linreg_rsi(price=None,linreg_period=160,ma_period=120,
                    rsi_period=150,trade='long'):

    """
        function to calculate entries point using special linear regression & macd indicator
        
        Args :
            price (Series): input price like open, high, low & close price etc
            ma_period (Series): ma period to calculate wma
            linreg_period (Series): period to calculate moving linear regression
            rsi_period (Series): to calculate the rsi value
            trade (str): trade to identify poaition is long or short
        
        Retrun :
            entries (Series): entries signals
    """

    # Defining INDICATORS to be use
    # Calculate RSI crossover with MA of RSI
    vbt.talib('RSI')
    rsi=vbt.RSI.run(price,window=rsi_period)
    vbt.talib('MA')
    rsi_ma=vbt.MA.run(rsi.rsi,window=ma_period)
    # Calculate Price above moving LinReg 
    linreg=signal.LinReg(price,period=linreg_period)

    # Defining Rules
    if trade=='long':
        en1=rsi.rsi.vbt.crossed_above(rsi_ma.ma)
        en2=price > linreg
    elif trade=='short':
        en1=rsi.rsi.vbt.crossed_below(rsi_ma.ma)
        en2=price < linreg
    

    # Defining Entries logic
    entries=signal.combine_entries(en1,en2)
    return entries


def macd(price=None,fast=12,slow=26,signal_period=9,trade='long'):

    macd, macdsignal, macdhist = talib.MACDEXT(price, fastperiod=12, fastmatype=0, 
                                            slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    
    if trade=='long':
        entries=macd.vbt.crossed_above(macdsignal)
    elif trade=='short':
        entries=macd.vbt.crossed_below(macdsignal)

    return entries

@jit
def get_bollinger_bands(prices, rate=20,std_num=2):
    lin=talib.LINEARREG(prices,timeperiod=rate).to_numpy()
    std = prices.rolling(rate).std()
    bollinger_up = lin + std * std_num # Calculate top band
    bollinger_down = lin - std * std_num # Calculate bottom band
    return bollinger_up, lin ,bollinger_down

@jit
def LBBANDS(close=None,period=20,std=2):
    upper, middle, lower = get_bollinger_bands(prices=close,rate=period,std_num=std)
    return upper, middle, lower

def trend_long(open, high, low, close, volume,period=40):
    # Volume Indicator
    ad = vbt.talib("AD").run(high, low, close, volume).real.to_numpy()
    adosc = vbt.talib("ADOSC").run(high, low, close, volume).real.to_numpy()
    obv = vbt.talib("OBV").run(close, volume).real

    # Momentum Indicator
    # SMA,MA_EMA,MA_WMA,MA_DEMA,MA_TEMA,MA_TRIMA,MA_KAMA,MA_MAMA,MA_T3
    sma = vbt.talib("SMA").run(close, timeperiod=60).real.to_numpy()
    dema = vbt.talib("DEMA").run(close, timeperiod=50).real.to_numpy()
    tema = vbt.talib("TEMA").run(close, timeperiod=50).real.to_numpy()
    trima = vbt.talib("TRIMA").run(close, timeperiod=50).real.to_numpy() 
    kama = vbt.talib("KAMA").run(close, timeperiod=50).real.to_numpy()
    rsi = vbt.talib("RSI").run(close, timeperiod=40).real.to_numpy()
    adx = vbt.talib("ADX").run(high, low, close, timeperiod=14).real.to_numpy()
    bop= vbt.talib("BOP").run(open, high, low, close).real.to_numpy()
    sar= vbt.talib("SAR").run(high, low).real.to_numpy()
    rocr= vbt.talib("ROCR").run(close).real.to_numpy()
    ultosc= vbt.talib("ULTOSC").run(high, low, close).real.to_numpy()
    mfi = vbt.talib("MFI").run(high, low, close, volume).real.to_numpy()    
    ema = vbt.talib("EMA").run(close, timeperiod=3).real
    wma = vbt.talib("WMA").run(close, timeperiod=50).real
    cci= vbt.talib("CCI").run(high, low, close, timeperiod=14).real

    mama = vbt.talib("MAMA").run(close)
    stoch = vbt.talib("STOCH").run(high, low, close, 40)
    stochf = vbt.talib("STOCHF").run(high, low, close, 40)
    stochrsi = vbt.talib("STOCHRSI").run(high, 40)
    macd = vbt.talib("MACD").run(close)
    macdext = vbt.talib("MACDEXT").run(close)
    bbands = vbt.talib("BBANDS").run(close,timeperiod=21)

    # Volatility Indicators
    atr= vbt.talib("ATR").run(high, low, close).real.to_numpy()

    # Statistic Functions
    linreg = vbt.talib("LINEARREG").run(close, 40).real.to_numpy()
    linearreg_slop=vbt.talib("LINEARREG_SLOPE").run(close).real.to_numpy()
    stddev=vbt.talib("STDDEV").run(close).real

    # Custom Indictor
    # 4 EMA ( put 3 - 21- 55 - 89  ) 
    ema2=vbt.talib("EMA").run(ema, timeperiod=21).real
    ema3=vbt.talib("EMA").run(ema2, timeperiod=55).real
    ema4=vbt.talib("EMA").run(ema3, timeperiod=89).real
    # 2 BBands ( put 21 - 89 )
    bbands2 = vbt.talib("BBANDS").run(bbands.middleband,timeperiod=89)
    # 2 WMA
    wma2 = vbt.talib("WMA").run(wma, timeperiod=50).real
    # 2 MACDEXT
    macdext2 = vbt.talib("MACDEXT").run(macdext.macd)
    # 2 OBV
    # obv = vbt.talib("OBV").run(close, volume).real
    # 4 STDDEV ( of something we will put them on )
    stddev2=vbt.talib("STDDEV").run(stddev).real
    stddev3=vbt.talib("STDDEV").run(stddev2).real
    stddev4=vbt.talib("STDDEV").run(stddev3).real


    return  ad, adosc, obv, sma, dema, tema, trima,\
            kama, rsi, adx, bop, sar, rocr, ultosc, mfi, ema,\
            wma, cci, atr, linreg, linearreg_slop, stddev, ema4,\
            wma2, stddev4, mama.mama, mama.fama

def indicator():
    ind = vbt.IndicatorFactory(
    class_name = "Combination_Trend_Long",
    short_name = "trend_long",
    input_names  = ["open", "high", "low", "close", "volume"],
    param_names = ["period"],
    output_names = [ "ad", "adosc", "obv", "sma", "dema", "tema", "trima", "kama", 
                     "rsi", "adx", "bop", "sar", "rocr", "ultosc", "mfi", "ema", 
                     "wma", "cci", "atr", "linreg", "linearreg_slop", "stddev", "ema4",
                     "wma2", "stddev4", "mama", "fama"],
        ).with_apply_func(
            # trend_chaser, 
            trend_long,
            keep_pd=True, 
            takes_1d=True
    )
    return ind