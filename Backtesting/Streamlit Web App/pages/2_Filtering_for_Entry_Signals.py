# Import Libraries
import sys
sys.path.append('../')
import streamlit as st
import pandas as pd
import Harmo.signal as signal
from Signal_Creation import SIGNALS, exit_strategies
import numpy as np
from datetime import datetime
import os

path = os.getcwd().split('Harmolight')[0]+'Harmolight\MAIN DATA SAVING'

# Remove streamlit form design
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''
st.markdown(css, unsafe_allow_html=True)


if 'data_verified' not in st.session_state:
    st.session_state['data_verified'] = False

if 'percent' not in st.session_state:
    st.session_state['percent'] = 1.0


# Class for loading data from txt or csv file and convert to dataframe
class DATA_LOADER():
    def read_dataset(self, uploaded_file=None):
        """
            read_dataset function use to read the dataset file from the
            streamlit file uploading widget

            Input :
                uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): uploaded file of dataset
            
            output :
                df (Dataframe): dataframe of dataset

        """
        df = signal.read_file(uploaded_file)
        if df is not None:
            df = df[~df.index.duplicated(keep='first')]
        return df
    
    def read_signal(self, uploaded_file=None):
        """
            read_signal function use to read the signal file from the
            streamlit file uploading widget

            Input :
                uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): uploaded file of signals
            
            output :
                df (Dataframe): dataframe of entry & exit signals

        """
        df = pd.read_csv(uploaded_file)
        columns = df.columns.to_list()
        if set(['Entry','Exit','Open time']).issubset(columns): 
            df.set_index(pd.to_datetime(df['Open time']), inplace=True)
            df = df.drop("Open time", axis=1)
            df = df[~df.index.duplicated(keep='first')]
        else:
            df = None

        return df


# Filters class to apply filters on data
class FILTERS():
    # function for filtering of lossing trades
    def lossing_trade_filter(self, df=None, flt_check=False):
        """
            lossing_trade_filter function use to filter out lossing trades

            Input:
                df (DataFrame): dataframe having data of trades
                flt_check (bool): variable use to tell do we need to filter lossing trades or not

            Output:
                df (DataFrame): dataframe having data of wining trades
        
        """
        if flt_check:
            df['Entry'] = np.where((df['status copy'] == "Win"), True, False)
            df['Exit'] = np.where((df['Trade Status (Win/Loss)'] == "Win"), True, False)
        
        return df
    

    # function to filter trades between specific time range
    def time_filter(self, df=None, start_time='00:00', end_time='23:59'):
        """
            time_filter function use to filter out trades between specific time range

            Input:
                df (DataFrame): dataframe having having data of trades
                flt_check (bool): variable use to tell do we need to filter lossing trades or not

            Output:
                df (DataFrame): dataframe having data of trades
        
        """

        if start_time !='00:00' or end_time!='23:59':
            # create new column to get time
            df['time'] = pd.DatetimeIndex(df.index)
            df['time'] = df['time'].dt.time

            # get indexes of all trades present b/w specific time range
            if start_time<end_time:
                idx = np.where((df['time'] > start_time) & (df['time'] < end_time),True, False)
            elif end_time<start_time:
                idx = np.where((df['time'] > start_time) | (df['time'] < end_time), True, False)
            
            # filter out all indexes not present in define time range
            df['Entry'][~idx] = False
            df['Exit'][~idx] = False
            df.drop(columns=['time'], inplace=True)

        return df
    

# class object & variables to apply different checks
obj = SIGNALS()
read_obj = DATA_LOADER() 
flt_obj = FILTERS()
mf_check = False
x_y_bar = False


# exit signal function
@st.cache
def exit_signal(data_df, entries, exit_strategy, trade_type, value, rr):
    signal_df = obj.exit_signals(data_df, entries, exit_strategy, trade_type, value, rr)
    signal_df = signal_df[['Entry', 'Entry Price', 'Exit', 'Exit Price', 
                           'Trade Compeletion Days', 'Trade Status (Win/Loss)']]
    return signal_df


# load data file
@st.cache(allow_output_mutation=True)
def read_data(uploaded_file_data):
    data_df = signal.read_file(uploaded_file_data)
    return data_df


# load signals file
@st.cache(allow_output_mutation=True)
def read_signal(uploaded_file):
    signal_df = pd.read_csv(uploaded_file)
    signal_df.set_index(pd.to_datetime(signal_df['Open time']), inplace=True)
    signal_df = signal_df.drop("Open time", axis=1)
    return signal_df

# draw candlestic chart with entry & exit signals
@st.cache(allow_output_mutation=True)
def figures(data_df, signal_df, percent = 0.05):
    last_percent = int(len(data_df) * (percent - 0.05))
    percent = int(len(data_df) * percent)
    data_df = data_df.iloc[last_percent : percent]
    signal_df = signal_df.iloc[last_percent : percent]
    fig = data_df.vbt.ohlcv.plot()
    try:
        signal_df['Entry'].vbt.signals.plot_as_entries(signal_df['Entry Price'], fig=fig)
        signal_df['Exit'].vbt.signals.plot_as_exits(signal_df['Exit Price'], fig=fig)
    except:
        signal_df['Entry'].vbt.signals.plot_as_entries(signal_df['Entry Price'], fig=fig)
    fig.update_layout(
        yaxis_title="Price ($)",
        hovermode="x unified",
    )
    return fig


# function to return bars count b/w entry and exit
@st.cache
def bars_count_fun(temp_df):
    # get indexes of entry & exit signals
    exit_idx = temp_df.index[temp_df['Exit']==True]
    entry_idx = temp_df.index[temp_df['Entry']==True]
    
    # filter any extra entry
    if len(exit_idx) < len(entry_idx):
        filter_entries = temp_df['Entry'][temp_df['Entry']==True]
        filter_entries[-1] = False
        temp_df['Entry'][temp_df['Entry']==True] = filter_entries
        entry_idx = entry_idx[:len(exit_idx)]
    
    # calculate the bar between each trade
    bars_count = temp_df.index.get_indexer(exit_idx) - temp_df.index.get_indexer(entry_idx)
    return bars_count


# function to save data into specific location with timestamp 
def create_file(data_df=None, temp_df=None):
    # create copy of dataframe and add target column in it
    mf_df = data_df.copy()
    mf_df['Target'] = np.array(temp_df['Entry'])
    
    # name of file with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "step2_data_{}.csv".format(timestamp)
    
    # save file specific folder if folder not present then first create it & then save file
    if os.path.isdir(path):
        mf_df.to_csv(path+"/"+filename)
        st.success(path+'\\' + filename)
    else:
        st.error("MAIN DATA SAVING folder not exist")


# title of Page
st.title("Step2: Filter Signals")
st.markdown("___")

# sections
tab1, tab2 = st.tabs(["Filter Signals", "Generate Exit Signals"])

try:
    with tab1:
        header = st.container()
        filters = st.container()
        download = st.container()
        
        # Data & Signal Files Verfication Section 
        with header:    
            # Widgets to upload signals & orignal datasets 
            uploaded_file_data = st.file_uploader("Select Data File", type=['CSV','TXT'])
            uploaded_file = st.file_uploader("Select Signals File", type=['CSV','TXT'])
            
            # Data & Signal Files Verfication Part
            if (uploaded_file_data is not None) and (uploaded_file is not None):
                # read files
                data_df = read_obj.read_dataset(uploaded_file=uploaded_file_data)
                signal_df = read_obj.read_signal(uploaded_file=uploaded_file)

                # apply checks on files for verification
                if data_df is not None:
                    if signal_df is not None:
                        if data_df.index.equals(signal_df.index):
                            # add heading on files
                            st.subheader("Dataset Table")
                            st.write(data_df.head())
                            st.subheader("Signals Table")
                            st.write(signal_df.head())
                            st.session_state['data_verified'] = True
                        else:
                            st.error("Your signal & dataset files not Matching. Please use right files. Thank You!")
                    else:
                        st.error("Signal File not correct. Please select correct Signal File. Thank You!")
                else:
                    st.error("Data File not correct. Please select correct Data File. Thank You!")

        # filters section
        with filters:
            # check files upload or not
            if st.session_state['data_verified']:         
                # Candlestic Chart Plot
                data_samples = [str(i)+"% - "+str(i+5)+"%" for i in range(0,100,5)]
                options = {}
                dsize = 0.05
                for i in data_samples:
                    options[i] = dsize
                    dsize += 0.05
                val = st.selectbox("Data Portion",options, index = 9)
                st.session_state['percent'] = options[val]

                st.subheader("Trades Chart")
                st.plotly_chart(figures(data_df, signal_df, percent=st.session_state['percent']))

                # Filters
                with st.form(key='my-form'):
                    
                    # filters for add or delete entry signals widgets
                    last_percent = int(len(data_df) * (st.session_state['percent'] - 0.05))
                    percent = int(len(data_df) * st.session_state['percent'])
                    # inds= st.multiselect("Select Multiple Index Positions", data_df.iloc[last_percent:percent].index)
                    # f1_col, f2_col, f3_col = st.columns(3)
                    # add_signal = f1_col.checkbox("Add Signal")
                    # delete_signal = f2_col.checkbox("Delete Signal")

                    # filters widgets
                    st.subheader("Filters")
                    first_col, sec_col,third_col = st.columns(3)
                    start_time = first_col.time_input("Start Time", value=datetime.strptime('00:00', '%H:%M'))
                    end_time = first_col.time_input("End Time", value=datetime.strptime('23:59', '%H:%M'))
                    x_bars = sec_col.number_input("X Number of Bars Before Entry Point", min_value=0)
                    y_bars = sec_col.number_input("Y Number of Bars After Entry Point", min_value=0)
                    bars = third_col.number_input("Max Trading Time in Number of Bars", min_value=0)    
                    lossing_trades = first_col.checkbox("Filter Lossing Trades")
                    save_file = sec_col.checkbox("Save File")
                    
                    # submit values to apply filters
                    if st.form_submit_button("Submit"):                    
                        # ####   Load Files   ####
                        signal_df['status copy'] = np.where((signal_df['Entry']==True) | (signal_df['Exit']==True),signal_df['Trade Status (Win/Loss)'],"Nothing")
                        signal_df['status copy'] = signal_df['status copy'][(signal_df['Entry']==True) | (signal_df['Exit']==True)].shift(-1)
                        temp_df = signal_df.copy()

                        # ####   Filter wining trades   ####  
                        temp_df = flt_obj.lossing_trade_filter(df=temp_df, flt_check=lossing_trades)

                        # ####   Filter Bars   ####  
                        if bars > 0:                  
                            bars_count = bars_count_fun(temp_df)
                            
                            # initialize bars column & store no. of bars duration between each trade
                            temp_df['bars'] = 0
                            temp_df['bars'][temp_df['Exit']==True] = bars_count
                            temp_df['bars'][temp_df['Entry']==True] = bars_count

                            # Entry and exit signals
                            temp_df['Entry'] = np.where((temp_df['Entry']==True) & (temp_df['bars']<bars),True,False)
                            temp_df['Exit'] = np.where((temp_df['Exit']==True) & (temp_df['bars']<bars),True,False)

                        # # ####   Filter Time   ####
                        temp_df = flt_obj.time_filter(df=temp_df, start_time=start_time, end_time=end_time)

                        # # ####  Add Signals  ####
                        # if add_signal:
                        #     temp_df['Entry'].loc[inds] = True
                        #     temp_df['Entry Price'].loc[inds] = data_df['Close'].loc[inds]

                        # # ####  Delete Signals  ####
                        # if delete_signal:
                        #     temp_df['Entry'].loc[inds] = False
                        #     temp_df['Entry Price'].loc[inds] = 0

                        # Filter X & Y amount of bars before and after entry position
                        # if x_bars>0 or y_bars>0:
                        #     idx = temp_df.index.get_indexer(temp_df[temp_df['Entry']==True].index)
                        #     y_idx = (idx-x_bars , idx+y_bars+1)
                        #     df_list = [temp_df.iloc[i:j] for i,j in zip(y_idx[0],y_idx[1])]
                        #     flt_df = pd.concat(df_list, axis=0)
                        #     flt_df = flt_df.drop_duplicates()
                        try:
                            # Filter X & Y amount of bars before and after entry position
                            if x_bars>0 or y_bars>0:
                                idx = temp_df.index.get_indexer(temp_df.index[temp_df['Entry']==True])
                                y_idx = (idx-x_bars , idx+y_bars+1)
                                df_list = [temp_df.iloc[i:j] for i,j in zip(y_idx[0],y_idx[1])]
                                # print(df_list)
                                flt_df = pd.concat(df_list, axis=0)
                                flt_df = flt_df.reset_index().drop_duplicates().set_index('Open time')
                                # flt_df = flt_df.drop_duplicates()  
                                print(flt_df) 
                            
                            # mf_check = True
                            # st.session_state['signal_df'] = signal_df 
                            # st.plotly_chart(figures(data_df, signal_df, percent=st.session_state['percent']))              
                        except Exception as e:
                            st.error(str(e)+". No entry signal present!\nPlease add some entry signals first.")


                        mf_check = True
                        st.plotly_chart(figures(data_df, temp_df, percent=st.session_state['percent']))

        # Downloading section
        with download:
            # Check file create or not. if created then give option to download it 
            if mf_check and save_file:
                try:
                    create_file(data_df.loc[flt_df.index], flt_df)
                except:
                    create_file(data_df, temp_df)
                    
except: 
    st.error("Files Not Correct. Please use right files. Thank You!")

with tab2:
    # Widgets to upload signals & orignal datasets 
    uploaded_file_data = st.file_uploader("Select Dataset File", type=['CSV','TXT'])
    uploaded_file = st.file_uploader("Select Signals Dataset File", type=['CSV','TXT'])
    try:
        # check dataset files or not
        if (uploaded_file_data is not None) and (uploaded_file is not None):
            data_df = read_data(uploaded_file_data)
            signal_df = read_signal(uploaded_file)
            if data_df.index.equals(signal_df.index):
                # add heading on files
                st.subheader("Dataset Table")
                st.write(data_df.head())
                st.subheader("Signals Table")
                st.write(signal_df.head())
                
                # check signals file if it has exit signal or not 
                if 'Entry' in signal_df.columns.to_list():
                    # select long & short entry and exit
                    trade_type = st.checkbox("Short Logic")
                    exit_strategy = st.selectbox("Select Exit Strategy", exit_strategies)
                    
                    # Select SL or x no. of bars etc according to selected strategy
                    if exit_strategy == "Fixed SL Exit":
                        value = st.number_input("Select SL Amount", min_value=0.1, step=0.5, value=1.0)
                        rr = st.number_input("Select Risk Reward Ratio",min_value=0.1, max_value=100.0, step=0.5, value=1.5)
                    elif exit_strategy == 'Last X Bars SL Exit':
                        value = st.number_input('Select Last X Number of Bars for SL', min_value=1, step=1, value=1)
                        rr = st.number_input("Select Risk Reward Ratio",min_value=0.1, max_value=100.0, step=0.5, value=1.5)
                    elif exit_strategy == 'Fixed Bar Exit':
                        value = st.number_input('Select X Number of Bars', min_value=1, step=1, value=1)
                        
                    if st.button("Generate Signals"):
                        signal_df = exit_signal(data_df, signal_df['Entry'], exit_strategy, trade_type, value, rr)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = "step2_signals_dataset_{}.csv".format(timestamp)
                        signal_df.to_csv(path+"/"+filename)
                        st.success('File saved at this location. ' + os.getcwd()+'\\generated_files\\' + filename)
                else:
                    st.error("Your File don't have any entry column. Please select correct file. Thank You!")
            else:
                st.error("Your signal & dataset files not Matching. Please use right files. Thank You!")
    except:
        st.error("Files Not Correct. Please use right files. Thank You!")