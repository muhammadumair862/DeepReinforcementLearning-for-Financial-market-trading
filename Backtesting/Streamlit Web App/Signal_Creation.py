# Import Libraries
import sys
sys.path.append('../')
import streamlit as st
import Harmo.strategy as signal
import os
from datetime import datetime

path = os.getcwd().split('Harmolight')[0]+'Harmolight\MAIN DATA SAVING'

# Entry & Exit Strategies
exit_strategies = ['Fixed SL Exit', 'Last X Bars SL Exit', 'Fixed Bar Exit']
entry_strategies = ['MACD Entry', 'X Bar Entry']


# Class to create read & generate signals
class SIGNALS():
    def __init__(self, df=None, timeframe='default') -> None:
        df = df
        timeframe = timeframe

    @st.cache(allow_output_mutation=True)
    def read_file(self, uploaded_file):
        df = signal.read_file(uploaded_file)
        return df        

    def entry_signal(self, close_price, volume, entry_strategy, trade_type, x_bar):
        if trade_type:
            if entry_strategy=='MACD Entry':
                entries = signal.sell_indicator(close_price)
        else:
            if entry_strategy=='MACD Entry':
                entries = signal.macd_entry(close_price, volume)
            elif entry_strategy=='X Bar Entry':
                entries = signal.x_bar_entry(close_price, x_bar=x_bar)
                
        return entries

    def exit_signals(self, df=None, entries=None, strategy=None, trade_type=None, value=None, RR=1.5):
        if trade_type:
            if strategy=='Last X Bars SL Exit':
                df = signal.exit_shortsignal(df, entries=entries, Num_Bars=value, rr=RR)
            elif strategy=='Fixed SL Exit':
                df = signal.fixed_exit_shortsignal(df, entries=entries, SL=value, RR=RR)
            elif strategy=='Fixed Bar Exit':
                df = signal.bar_base_shortexit(df, entries=entries, num_bars=value)        
        else:
            if strategy=='Last X Bars SL Exit':
                df = signal.exit_signal_1(df, entries=entries, Num_Bars=value, rr=RR)
            elif strategy=='Fixed SL Exit':
                df = signal.fixed_exit_signal_1(df, entries=entries, SL=value, RR=RR)
            elif strategy=='Fixed Bar Exit':
                df = signal.bar_base_exit_1(df, entries=entries, num_bars=value)
        return df

    def signals(self, df=None, entry_strategy=None, exit_strategy=None, trade_type=None, value=None, rr=None, x_bar=5):
        entries = self.entry_signal(df['Close'],df['Volume'], entry_strategy, trade_type, x_bar)
        df = self.exit_signals(df, entries, exit_strategy, trade_type, value, rr)
        return df[['Entry','Entry Price','Exit','Exit Price','Trade Compeletion Days','Trade Status (Win/Loss)']]


# To stop data loading every time when any widget update on interface
@st.cache
def load_signal(df, entry_strategy, exit_strategy, trade_type, value, rr=1.5, x_bar_val=5):
    obj = SIGNALS()   # object of signal class  
    df = obj.signals(df, entry_strategy, exit_strategy, trade_type, value, rr, x_bar_val)
    return df


# Create object of class
obj = SIGNALS()

# check to see signals file generate or not
sf_check = False   

# title of Page
st.title("Step1: Create Signals")
st.markdown("___")

# Sections
header = st.container()
download = st.container()

# Main section
with header:
    # Widgets for upload file
    uploaded_file = st.file_uploader("Select Data", type=["txt", "csv"])
    if uploaded_file is not None:
        df = obj.read_file(uploaded_file)
        st.write(df)

        # This part work when user input correct dataset
        if df is not None:
            # select long & short entry and exit
            trade_type = st.checkbox("Short Logic")
            first_col, sec_col = st.columns(2)
            
            # Widgets for select entry & exit strategy
            entry_strategy = first_col.selectbox("Select Entry Strategy", entry_strategies)
            exit_strategy = first_col.selectbox("Select Exit Strategy", exit_strategies)

            # Select x no. of bars etc according to selected entry strategy
            if entry_strategy == 'X Bar Entry':
                entry_value = sec_col.number_input("X number of Bars", min_value=1, step=1, value=5)
                
            # Select SL or x no. of bars etc according to selected strategy
            if exit_strategy == "Fixed SL Exit":
                value = sec_col.number_input("Select SL Amount", min_value=0.1, step=0.5, value=1.0)
                rr = sec_col.number_input("Select Risk Reward Ratio",min_value=0.1, max_value=100.0, step=0.5, value=1.5)
            elif exit_strategy == 'Last X Bars SL Exit':
                value = sec_col.number_input('Select Last X Number of Bars for SL', min_value=1, step=1, value=1)
                rr = sec_col.number_input("Select Risk Reward Ratio",min_value=0.1, max_value=100.0, step=0.5, value=1.5)
            elif exit_strategy == 'Fixed Bar Exit':
                value = sec_col.number_input('Select X Number of Bars', min_value=1, step=1, value=1)
                                    
            # Submit details and generate signals
            if uploaded_file and st.button("Submit"):
                if entry_strategy == 'X Bar Entry':
                    if exit_strategy == 'Fixed Bar Exit':
                        signal_df = load_signal(df, entry_strategy, exit_strategy, trade_type, value, x_bar_val=entry_value)
                    else:
                        signal_df = load_signal(df, entry_strategy, exit_strategy, trade_type, value, rr, entry_value)
                else:
                    if exit_strategy == 'Fixed Bar Exit':
                        signal_df = load_signal(df, entry_strategy, exit_strategy, trade_type, value)
                    else:
                        signal_df = load_signal(df, entry_strategy, exit_strategy, trade_type, value, rr)
                
                st.write(signal_df)
                sf_check = True
        else:
            st.error("File incorrect. Please use correct file. Thank You")

# Downloading section
with download:
    # Check file create or not. if created then give option to download it 
    if sf_check:
        # name of file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "step1_signals_{}.csv".format(timestamp)

        # save file specific folder if folder not present then first create it & then save file
        if os.path.isdir(path):
            signal_df.to_csv(path+"/"+filename)
            st.success(path+'\\' + filename)
        else:
            st.error("MAIN DATA SAVING folder not exist")