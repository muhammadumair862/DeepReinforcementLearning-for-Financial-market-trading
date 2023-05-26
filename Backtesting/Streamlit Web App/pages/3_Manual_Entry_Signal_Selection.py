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

css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''
st.markdown(css, unsafe_allow_html=True)


if 'percent' not in st.session_state:
    st.session_state['percent'] = 1.0
# class object & variables to apply different checks
obj = SIGNALS()
mf_check = False
file_check = False


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
    # data_df = signal.read_file(uploaded_file_data)
    data = pd.read_csv(uploaded_file_data)
    
    # preprocess columns names and create datetime index
    data.columns = [i.strip() for i in data.columns] 
    data['Open time'] = pd.to_datetime(data['Date'] + data['Time'])
    data.index = data['Open time']
    data.rename(columns={'Last':'Close'},inplace=True)
    data.drop(columns=['Open time','Date','Time'], inplace=True)
    
    return data


# load signals file
@st.cache(allow_output_mutation=True)
def man_read_signal(uploaded_file):
    signal_df = pd.read_csv(uploaded_file)
    signal_df.set_index(pd.to_datetime(signal_df['Open time']), inplace=True)
    signal_df = signal_df.drop("Open time", axis=1)
    signal_df['Entry Price'] = np.where(signal_df['Entry']==True, signal_df['Close'], 0)
    return signal_df

# load signals file
@st.cache(allow_output_mutation=True)
def read_empty_signal():
    signal_df = pd.DataFrame(data={'Entry':[False],'Exit':[False],'Entry Price':[0],'Exit Price':[0]}, index=data_df.index)
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


# function to save data into specific location with timestamp 
def create_file(data_df=None, temp_df=None):
    # create copy of dataframe and add target column in it
    mf_df = data_df.copy()
    mf_df['Target'] = np.array(temp_df['Entry'])
    print("working",os.getcwd())
    # name of file with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "step3_data_{}.csv".format(timestamp)
    
    # save file specific folder if folder not present then first create it & then save file
    if os.path.isdir(path):
        mf_df.to_csv(path+"/"+filename)
        st.success(path+'\\' + filename)
    else:
        st.error("MAIN DATA SAVING folder not exist")


# title of Page
st.title("Step3: Label Entry Signals")
st.markdown("___")

try:
    header = st.container()
    filters = st.container()
    download = st.container()

    # header section
    with header:
        # Widgets to upload signals & orignal datasets 
        uploaded_file_data = st.file_uploader("Select Data File", type=['CSV','TXT'], key='manual-dt-load')
        # uploaded_file = st.file_uploader("Select Signals File", type=['CSV','TXT'], key='manual-sig-load')
        
        # check dataset files or not
        if (uploaded_file_data is not None):
            data_df = read_data(uploaded_file_data)
            if data_df is not None:
                data_df = data_df[~data_df.index.duplicated(keep='first')]
                if 'signal_df' not in st.session_state:
                    # if uploaded_file is not None :
                    #     signal_df = man_read_signal(uploaded_file)
                    # else:
                    #     signal_df = read_empty_signal()
                    signal_df = read_empty_signal()
                    signal_df = signal_df[~signal_df.index.duplicated(keep='first')]
                    st.session_state['signal_df'] = signal_df
                else:
                    signal_df = st.session_state['signal_df']
                    signal_df = signal_df[~signal_df.index.duplicated(keep='first')]
                file_check = True

                # if data_df.index.equals(signal_df.index):
                #     # add heading on files
                #     st.subheader("Dataset Table")
                #     st.write(data_df.head())
                #     st.subheader("Signals Table")
                #     st.write(signal_df.head())
                # else:
                #     st.error("Your signal & dataset files not Matching. Please use right files. Thank You!")
            else:
                st.error("Data File not correct. Please select correct Data File. Thank You!")


    # check files upload or not
    if file_check:         
        # filters section
        with filters:
            data_samples = [str(i)+"% - "+str(i+5)+"%" for i in range(0,100,5)]
            options = {}
            dsize = 0.05
            for i in data_samples:
                options[i] = dsize
                dsize += 0.05
            val = st.selectbox("Data Portion",options, index = 9)
            st.session_state['percent'] = options[val]

            # st.session_state['percent'] = st.slider("part of data", min_value=0.1, max_value=0.9, step=0.1, value=st.session_state['percent'])

            st.subheader("Trades Chart")
            st.plotly_chart(figures(data_df, signal_df, percent=st.session_state['percent']))

            with st.form(key='my-form'):
                last_percent = int(len(data_df) * (st.session_state['percent'] - 0.05))
                percent = int(len(data_df) * st.session_state['percent'])
                inds= st.multiselect("Multiple Index Positions", data_df.iloc[last_percent:percent].index)
                f1_col, f2_col, f3_col = st.columns(3)
                add_signal = f1_col.checkbox("Add Signal")
                delete_signal = f2_col.checkbox("Delete Signal")
                delete_signals = f3_col.checkbox("Delete All Signal")

                # filters
                st.subheader("Filters")
                first_col, sec_col = st.columns(2)
                start_time = first_col.time_input("Start Time", value=datetime.strptime('00:00', '%H:%M'))
                end_time = first_col.time_input("End Time", value=datetime.strptime('23:59', '%H:%M'))
                x_bars = sec_col.number_input("X Number of Bars Before Entry Point", min_value=0)
                y_bars = sec_col.number_input("Y Number of Bars After Entry Point", min_value=0)
                save_file = first_col.checkbox("Save File")

                if st.form_submit_button("Submit"):  
                    # filters
                    # save_file = st.checkbox("Save File")
                    # ####  Add Signals  ####
                    # temp_df = signal_df.copy()
                    if add_signal:
                        signal_df['Entry'].loc[inds] = True
                        signal_df['Entry Price'].loc[inds] = data_df['Close'].loc[inds]


                    # ####  Delete Signals  ####
                    if delete_signal:
                        signal_df['Entry'].loc[inds] = False
                        signal_df['Entry Price'].loc[inds] = 0
                    
                    # ####  Delete Signals  ####
                    if delete_signals:
                        signal_df['Entry'] = False
                        signal_df['Entry Price'] = 0
                                    
                    # # ####   Filter Time   ####

                    if start_time and end_time:
                        temp_df = signal_df.copy()
                        temp_df['time'] = pd.DatetimeIndex(temp_df.index)
                        temp_df['time'] = temp_df['time'].dt.time
                        if start_time<end_time:
                            idx = np.where((temp_df['time'] > start_time) & (temp_df['time'] < end_time),True, False)
                        elif end_time<start_time:
                            idx = np.where((temp_df['time'] > start_time) | (temp_df['time'] < end_time), True, False)
                        signal_df['Entry'][~idx] = False
                        signal_df['Exit'][~idx] = False
                    try:
                        # Filter X & Y amount of bars before and after entry position
                        if x_bars>0 or y_bars>0:
                            idx = signal_df.index.get_indexer(signal_df.index[signal_df['Entry']==True])
                            y_idx = (idx-x_bars , idx+y_bars+1)
                            df_list = [signal_df.iloc[i:j] for i,j in zip(y_idx[0],y_idx[1])]
                            # print(df_list)
                            flt_df = pd.concat(df_list, axis=0)
                            flt_df = flt_df.reset_index().drop_duplicates().set_index('Open time')
                            # flt_df = flt_df.drop_duplicates()  
                            print(flt_df) 
                        
                        mf_check = True
                        st.session_state['signal_df'] = signal_df 
                        st.plotly_chart(figures(data_df, signal_df, percent=st.session_state['percent']))              
                    except Exception as e:
                        st.error(str(e)+". No entry signal present!\nPlease add some entry signals first.")
                        
                    

            # Downloading section
            with download: 
                # Check file create or not. if created then give option to download it 
                if mf_check and save_file:
                    try:
                        create_file(data_df.loc[flt_df.index], flt_df)
                    except:
                        create_file(data_df, signal_df)
except: 
    st.error("Files Not Correct. Please use right files. Thank You!")