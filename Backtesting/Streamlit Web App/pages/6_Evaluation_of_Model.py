# Import Libraries
import streamlit as st
import sys
sys.path.append('../')
import pandas as pd
import vectorbt as vbt
import Harmo.signal as signal
import Harmo.strategy as signal1
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
from Signal_Creation import SIGNALS, exit_strategies
import os
from datetime import datetime
import numpy as np

path = os.getcwd().split('Harmolight')[0]+'Harmolight\MAIN DATA SAVING'

css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''
st.markdown(css, unsafe_allow_html=True)


# initialize object
# obj = SIGNALS()
if 'stats_exists' not in st.session_state:
    st.session_state['stats_exists'] = False


# model predication function
def model_pred(model = None, df = None, features = None):
    X = preprocessing.scale(df[features])
    y_pred = model.predict(X)
    df['Entry'] = y_pred      # entries point store in dataframe
    return df


# function calculate portfolio using vectorbt
@st.cache(allow_output_mutation=True)
def result_cal(df = None, exit_strategy = None, value = None,
               rr = None, data_start = None, data_end =None,
               save_file = None, Fee=0.5, start_time='00:00', end_time='23:59'):   
    print(int((data_start/100)*len(df)),int((data_end/100)*len(df)))
    close = df['Close'].iloc[int((data_start/100)*len(df)):int((data_end/100)*len(df))] 
    df = df.iloc[int((data_start/100)*len(df)):int((data_end/100)*len(df))]
    # PORTOFOLIO PARAMETER
    Init_cash = 20000
    Size = 1

    # exits signals function
    if exit_strategy=='Last X Bars SL Exit':
        df1 = signal1.exit_signal_1(df.copy(), entries=df['Entry'], Num_Bars=value, rr=rr)
    elif exit_strategy=='Fixed SL Exit':
        df1 = signal1.fixed_exit_signal_1(df.copy(), entries=df['Entry'], SL=value, RR=rr)
    elif exit_strategy=='Fixed Bar Exit':
        df1 = signal1.bar_base_exit_1(df.copy(), entries=df['Entry'], num_bars=value) 
    # df1 = signal.exit_signal(df.copy(), df['Entry'], 1.2, 10)
    # df1 = exit_signal(df.copy(), df['Entry'], exit_strategy, value, rr)

    # # ####   Filter Time   ####
    if start_time and end_time:
        df1['time'] = pd.DatetimeIndex(df1.index)
        df1['time'] = df1['time'].dt.time
        if start_time<end_time:
            idx = np.where((df1['time'] > start_time) & (df1['time'] < end_time),True, False)
        elif end_time<start_time:
            idx = np.where((df1['time'] > start_time) | (df1['time'] < end_time), True, False)
        df1['Entry'][~idx] = False
        df1['Exit'][~idx] = False

    price = close.copy()
    price[df1['Entry']] = df1['Entry Price']
    price[df1['Exit']] = df1['Exit Price']
    
    en_df = df1[['Entry']][df1['Entry']==True]
    en_df = en_df.reset_index()
    # en_df.drop(columns=['index'], inplace=True)
    en_df.rename(columns={'Open time':'Entry time'}, inplace=True)
    ex_df = df1[['Exit','Trade Status (Win/Loss)']][df1['Exit']==True]
    ex_df = ex_df.reset_index()
    # ex_df.drop(columns=['index'], inplace=True)
    ex_df.rename(columns={'Open time':'Exit time'}, inplace=True)
    comb_df=pd.concat([en_df,ex_df], axis=1)
    comb_df.dropna(inplace=True)
    consecutive_win_list = []
    consecutive_loss_list = []
    consecutive_win_entry_list = []
    consecutive_loss_entry_list = []
    consecutive_win_exit_list = []
    consecutive_loss_exit_list = []
    count_win = 0
    count_loss = 0
    # iterate through the rows
    for i, row in comb_df.iterrows():
        if row['Trade Status (Win/Loss)'] == 'Win':
            count_loss=0
            count_win+=1
            if count_win==1:
                consecutive_win_entry_list.append(row['Entry time'])
                consecutive_win_exit_list.append(row['Exit time'])
                consecutive_win_list.append(count_win)
            else:
                consecutive_win_exit_list[len(consecutive_win_entry_list)-1] = row['Exit time']
                consecutive_win_list[len(consecutive_win_entry_list)-1] = count_win
        
        else:
            count_win=0
            count_loss+=1
            if count_loss==1:
                consecutive_loss_entry_list.append(row['Entry time'])
                consecutive_loss_exit_list.append(row['Exit time'])
                consecutive_loss_list.append(count_loss)
            else:
                consecutive_loss_exit_list[len(consecutive_loss_entry_list)-1] = row['Exit time']
                consecutive_loss_list[len(consecutive_loss_entry_list)-1] = count_loss
    win_df = pd.DataFrame({'Entry time':consecutive_win_entry_list,
                            'Exit time':consecutive_win_exit_list,
                            'Consecutive Wins':consecutive_win_list})
    win_df.sort_values(by=['Consecutive Wins'], inplace=True, ascending=False)
    
    loss_df = pd.DataFrame({'Entry time':consecutive_loss_entry_list,
                            'Exit time':consecutive_loss_exit_list,
                            'Consecutive Loss':consecutive_loss_list})
    loss_df.sort_values(by=['Consecutive Loss'], inplace=True, ascending=False)
    
    win_df = win_df[win_df['Consecutive Wins']>1]
    loss_df = loss_df[loss_df['Consecutive Loss']>1]

    # ex_df df1['Exit']==True]
    print(len(df1[df1['Entry']==True]), len(df1[df1['Exit']==True]))
    pf = vbt.Portfolio.from_signals(
        close=close, 
        entries=df1['Entry'], 
        exits=df1['Exit'],
        price=price,
        fixed_fees=Fee,
        size=Size,
        size_type='amount',
        init_cash=Init_cash
    )

    # save file 
    if save_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "step6_signals_{}.csv".format(timestamp)
        df1.to_csv(path+"/"+filename)
    return pf.stats(), win_df, loss_df,pf


def results_fun(model,save_file, win_df, loss_df, pf):
    col1, col2, col3, col4 = st.columns(4)
    starting_value = stats['Start Value']
    closing_value = round(float(stats['End Value']),2)
    total_return = round(float(stats['Total Return [%]']),4)
    win_trades = round(float(stats['Win Rate [%]']) * (float(stats['Total Trades'])/100),0)
    win_rate = round(float(stats['Win Rate [%]']),2)
    lossing_trades = int(stats['Total Trades']) - win_trades
    lossing_rate = round((100 - win_rate),2)

    # show statistics
    col1.metric("Start Balance", starting_value)
    col2.metric("Closing Balance", closing_value, str(total_return)+"%")
    col3.metric("Wining Trades", win_trades, str(win_rate)+"%")
    col4.metric("Lossing Trades", lossing_trades , str(-lossing_rate)+"%")

    st.subheader("Trades Chart")
    st.plotly_chart(pf.plot())
    st.table(stats)
    st.subheader("Consecutive Wins Trades Table")
    st.write(win_df)
    st.subheader("Consecutive Loss Trades Table")
    st.write(loss_df)
    if save_file:
        st.success('File saved at this location. ' + os.getcwd()+'\\generated_files')


# title of Page
st.title("Step6: Evaluation")
st.markdown("___")

# sections
header = st.container()
filters = st.container()
results = st.container()

# try:
with header:
    # widgets to get model and data
    uploaded_models = st.file_uploader("Select Model", type=['PKL'], accept_multiple_files=True)
    uploaded_file = st.file_uploader("Select Data File", type=['CSV','TXT'])

# with st.form(key='my-form'):
with filters:
    if len(uploaded_models) and (uploaded_file is not None):
        feature_set = []
        df = pd.read_csv(uploaded_file)  
        df.columns = [i.strip() for i in df.columns]
        print('Open time' not in df.columns.to_list())
        if 'Open time' not in df.columns.to_list():
            df['Open time'] = pd.to_datetime(df['Date'] + df['Time'])
            df.drop(columns=['Date','Time'], inplace=True)
        df.index = pd.to_datetime(df['Open time'])
        
        df.rename(columns={'Last':'Close'},inplace=True)
        df.drop(columns=['Open time'], inplace=True)         
        
        if os.path.isfile("model_log.csv"):
            log_file = pd.read_csv("model_log.csv")
            feature_set = df.columns.to_list()
            # feature_set = list(set(feature_set) - set(['Date', 'Time']))
            f_list = []
            for model in uploaded_models:
                f_list.append(log_file['Model Training Features'][log_file['Model Name']==model.name].iloc[0].split(","))
        else:
            feature_set = df.columns.to_list()
            default_feature_set = list(set(feature_set) - set(['Open','High','Low','Close','Open time', 'Date','Time']))       
        first_col, sec_col, third_col = st.columns(3)
        if all(x == f_list[0] for x in f_list):
            default_feature_set = f_list[0]
            if set(default_feature_set).issubset(set(feature_set)):
                # Widgets for select entry & exit strategy
                data_start = first_col.number_input("Select Starting Data for Training (%)", min_value=0, max_value=100, value=0)
                data_end = first_col.number_input("Select Ending Data for Training (%)", min_value=1, max_value=100, value=100)
                start_time = sec_col.time_input("Start Time", value=datetime.strptime('09:30', '%H:%M'))
                end_time = sec_col.time_input("End Time", value=datetime.strptime('16:00', '%H:%M'))
                fee = third_col.number_input("Commission Fee per Entry/Exit", min_value=0.0, value=0.75)        
                
                f1_col, f2_col, f3_col = st.columns(3)
                exit_strategy = f1_col.selectbox("Select Exit Strategy", exit_strategies)
                # Select SL or x no. of bars etc according to selected strategy
                if exit_strategy == "Fixed SL Exit":
                    value = f2_col.number_input("Select SL Amount", min_value=1, value=4)
                    rr = f3_col.number_input("Select Risk Reward Ratio", min_value=0.1, max_value=20.0, step=0.5, value=1.9)
                elif exit_strategy == 'Last X Bars SL Exit':
                    value = f2_col.number_input('Select Last X Number of Bars for SL', min_value=1,value=4)
                    rr = f3_col.number_input("Select Risk Reward Ratio", min_value=0.1, max_value=20.0, step=0.5, value=1.9)
                elif exit_strategy == 'Fixed Bar Exit':
                    value = f2_col.selectbox('Select X Number of Bars', range(1,1000))
                feature_set = st.multiselect("Select Input Features?", feature_set, default=default_feature_set)
                
                # st.multiselect("Select Exit Strategy", exit_strategies)
                # f2_col2, f3_col3 = st.columns(2)
                # f2_col2.number_input("Range of SL", min_value=0.1,step=0.1); f2_col2.number_input("", min_value=0.1, key='sl')
                # f3_col3.number_input("Range of Risk/Reward", min_value=0.1,step=0.1); f3_col3.number_input("", min_value=0.1, key='rr')
                
                save_file = f1_col.checkbox("Save File")
                

                # Submit details and generate signals
                if (data_start<data_end) and st.button("Submit"): 
                    # df = signal.read_file(uploaded_file)
                    for idx, uploaded_model in enumerate(uploaded_models):
                        model = joblib.load(uploaded_model)
                        # indicator_df = add_indicators(df)
                        # st.write(df)
                        try:
                            output_df = model_pred(model, df.copy(), feature_set)
                            if exit_strategy=='Fixed Bar Exit': 
                                stats, win_df, loss_df,pf = result_cal(df=output_df, exit_strategy=exit_strategy, value=value, 
                                                                    data_start=data_start, data_end=data_end, save_file=save_file, 
                                                                    Fee=fee, start_time=start_time, end_time=end_time)
                            else:
                                stats, win_df, loss_df, pf = result_cal(df=output_df, exit_strategy=exit_strategy, value=value, 
                                                                    rr=rr, data_start=data_start, data_end=data_end, save_file=save_file, 
                                                                    Fee=fee, start_time=start_time, end_time=end_time)
                            st.session_state['stats_exists'] = True
                            st.session_state['pf_'+str(idx)] = pf
                            st.session_state['stats_'+str(idx)] = stats
                            st.session_state['win_df_'+str(idx)] = win_df
                            st.session_state['loss_df_'+str(idx)] = loss_df
                        except:
                            st.error("Input Features invalied or more/less than required features")
                if data_end<=data_start:
                    st.error("Please select right part of data to train model.")
            else:
                st.error("Input data file not correct. Model train on these feature columns")
                st.error(default_feature_set)
                st.error("And your file have these feature columns")
                st.error(feature_set)
        else:
            st.error("Your models are not trained on same columns. So one use those column which trained on same columns.")
        

with results:
    if len(uploaded_models) and (uploaded_file is not None) and st.session_state['stats_exists']:
        st.subheader("Results")

        st.session_state['stats_exists'] = False
        tab1, tab2, tab3 = st.tabs(['Random Forest', 'AdaBoost', 'XGBoost'])
        # calculate statistics
        for i,model in enumerate(uploaded_models):
            if model.name.split('_')[0] == 'Random Forest':
                with tab1:
                    stats = st.session_state['stats_'+str(i)]
                    win_df = st.session_state['win_df_'+str(i)]
                    loss_df = st.session_state['loss_df_'+str(i)]
                    pf = st.session_state['pf_'+str(i)]
                    st.subheader(model.name)
                    results_fun(model, save_file, win_df, loss_df,pf)
            elif model.name.split('_')[0] == 'AdaBoost':
                with tab2:
                    stats = st.session_state['stats_'+str(i)]
                    win_df = st.session_state['win_df_'+str(i)]
                    loss_df = st.session_state['loss_df_'+str(i)]
                    pf = st.session_state['pf_'+str(i)]
                    st.subheader(model.name)
                    results_fun(model, save_file, win_df, loss_df,pf)
            elif model.name.split('_')[0] == 'XGBoost':
                with tab3:
                    stats = st.session_state['stats_'+str(i)]
                    win_df = st.session_state['win_df_'+str(i)]
                    loss_df = st.session_state['loss_df_'+str(i)]
                    pf = st.session_state['pf_'+str(i)]
                    st.subheader(model.name)
                    results_fun(model, save_file, win_df, loss_df,pf)
# except:
#     st.error("Files not Correct")