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
from itertools import product
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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


def get_high_low(df=None, sl=None, rr=None, window_size=5):
    n5 = pd.DataFrame({})
    n5 = df[['Open','High','Low','Close']]
    n5['next_5_bars_high'] = df['High'].rolling(window_size).max().shift(-(window_size-1))
    n5['next_5_bars_low'] = df['Low'].rolling(window_size).min().shift(-(window_size-1))
    n5['N_5'] = np.where(n5['High']==n5['next_5_bars_high'],1,
                        np.where(n5['Low']==n5['next_5_bars_low'],-1,
                        np.where(n5['High'].shift(-1)==n5['next_5_bars_high'],1,
                        np.where(n5['Low'].shift(-1)==n5['next_5_bars_low'],-1,
                        np.where(n5['High'].shift(-2)==n5['next_5_bars_high'],1,
                        np.where(n5['Low'].shift(-2)==n5['next_5_bars_low'],-1,
                        np.where(n5['High'].shift(-3)==n5['next_5_bars_high'],1,
                        np.where(n5['Low'].shift(-3)==n5['next_5_bars_low'],-1,
                        np.where(n5['High'].shift(-4)==n5['next_5_bars_high'],1,
                        np.where(n5['Low'].shift(-4)==n5['next_5_bars_low'],-1,
                        0))))))))))
    n5['next_10_bars_high'] = df['High'].rolling(window_size+5).max().shift(-(window_size+4))
    n5['next_10_bars_low'] = df['Low'].rolling(window_size+5).min().shift(-(window_size+4))
    n5['N_10'] = np.where(n5['High']==n5['next_10_bars_high'],1,
                        np.where(n5['Low']==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-1)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-1)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-2)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-2)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-3)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-3)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-4)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-4)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-5)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-5)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-6)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-6)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-7)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-7)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-8)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-8)==n5['next_10_bars_low'],-1,
                        np.where(n5['High'].shift(-9)==n5['next_10_bars_high'],1,
                        np.where(n5['Low'].shift(-9)==n5['next_10_bars_low'],-1,
                        0))))))))))))))))))))
    n5['next_15_bars_high'] = df['High'].rolling(window_size+10).max().shift(-(window_size+9))
    n5['next_15_bars_low'] = df['Low'].rolling(window_size+10).min().shift(-(window_size+9))
    n5['N_15'] = np.where(n5['High']==n5['next_15_bars_high'],1,
                        np.where(n5['Low']==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-1)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-1)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-2)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-2)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-3)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-3)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-4)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-4)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-5)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-5)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-6)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-6)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-7)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-7)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-8)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-8)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-9)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-9)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-10)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-10)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-11)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-11)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-12)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-12)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-13)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-13)==n5['next_15_bars_low'],-1,
                        np.where(n5['High'].shift(-14)==n5['next_15_bars_high'],1,
                        np.where(n5['Low'].shift(-14)==n5['next_15_bars_low'],-1,
                        0))))))))))))))))))))))))))))))
    n5.insert(0, 'SL', sl)
    n5.insert(1, 'RR', rr)

    return n5


def total_return_plot(df=None):
    # create a sample dataframe
    df['Backtest'] = range(1, len(df)+1)

    # create a line plot using Plotly Express
    fig = px.line(df, x='Backtest', y='Total Return [%]', hover_data=['SL', 'RR'])

    # highlight specific points
    fig.update_traces(mode='markers+lines', marker=dict(color='red'))
    fig.update_layout(title='Total Return Chart for Each Backtest')

    # display the plot in the Streamlit app
    st.plotly_chart(fig)


def HL_pct_plot(df=None):
    fig = make_subplots(rows=1, cols=3, specs=[[{'type':'pie'}, {'type':'pie'}, {'type':'pie'}]])

    fig.add_trace(go.Pie(labels=['High in Next 5 Bars', 'Low in Next 5 Bars'], values=[df['N_5'][df['N_5']==1].count(), df['N_5'][df['N_5']==-1].count()]), row=1, col=1)
    fig.add_trace(go.Pie(labels=['High in Next 10 Bars', 'Low in Next 10 Bars'], values=[df['N_10'][df['N_10']==1].count(), df['N_10'][df['N_10']==-1].count()]), row=1, col=2)
    fig.add_trace(go.Pie(labels=['High in Next 15 Bars', 'Low in Next 15 Bars'], values=[df['N_15'][df['N_15']==1].count(), df['N_15'][df['N_15']==-1].count()]), row=1, col=3)

    fig.update_layout(height=400, width=800, title_text="High & Low (%) in Next 5, 10, 15 Bars from Entry Position")
    # display the figure in the Streamlit app
    st.plotly_chart(fig)


def HL_plot(df=None):
    df['Timestamp'] = df.index

    # create a scatter plot for next_5_bars_high
    fig = px.scatter(df, x='Timestamp', y='next_5_bars_high', color_discrete_sequence=['green'])

    # add a scatter plot for next_5_bars_low, with a different color
    fig.add_trace(px.scatter(df, x='Timestamp', y='next_5_bars_low', color_discrete_sequence=['red']).data[0])

    # set the title and axis labels
    fig.update_layout(title='Next 5 Bars High vs Low', xaxis_title='Timestamp', yaxis_title='Price')

    # display the plot in the Streamlit app
    st.plotly_chart(fig)


# function calculate portfolio using vectorbt
# @st.cache(allow_output_mutation=True)
def multi_result_cal(df = None, exit_strategy = None, value = None,
                     rr = None, data_start = None, data_end =None,
                     save_file = None, Fee=0.5, start_time='00:00', end_time='23:59'):
       
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

    return pf.stats(), win_df, loss_df, df1

def multi_backtest(df = None, exit_strategy = None, value = None, value_end = None,
                     rr = None, rr_end = None, data_start = None, data_end =None,
                     save_file = None, Fee=0.5, n_top_results=20, start_time='00:00', end_time='23:59',
                     val_step = 0.1, rr_step = 0.1, model_name=None):
    
    if exit_strategy=='Fixed Bar Exit':
        th_combs = range(value, value_end, val_step)
    else:
        SL_val=np.arange(value,value_end,val_step)
        SL_val = np.round(SL_val, 2).tolist()
        RR_val=np.arange(rr,rr_end,rr_step)
        RR_val = np.round(RR_val, 2).tolist()
        th_combs = list(product(SL_val,RR_val))

    comb_stats=[]
    names=[]
    df_list=[]

    if exit_strategy=='Fixed Bar Exit':
        names = ['No. of bars']
        for sl in th_combs:
            result, _, _, result_df = multi_result_cal(df=df, exit_strategy=exit_strategy, value=sl,  
                            data_start=data_start, data_end=data_end, save_file=save_file, 
                            Fee=Fee, start_time=start_time, end_time=end_time)   
            comb_stats.append(result)        
    else:
        names = ['SL', 'RR']
        for sl,rr in th_combs:
            result, _, _, result_df = multi_result_cal(df=df, exit_strategy=exit_strategy, value=sl,rr=rr,  
                            data_start=data_start, data_end=data_end, save_file=save_file, 
                            Fee=Fee, start_time=start_time, end_time=end_time)  
            comb_stats.append(result) 
            indxes = result_df.index[result_df['Entry']==True]
            df_hl = get_high_low(df, sl, rr)

            df_list.append(df_hl.loc[indxes])       
    
    comb_stats_df = pd.DataFrame(comb_stats)
    try:
        comb_stats_df.insert(0, 'SL', [x[0] for x in th_combs])
        comb_stats_df.insert(1, 'RR', [x[1] for x in th_combs])
    except:
        comb_stats_df.insert(0, 'No. of bars', [x for x in th_combs])

    def sort_column(comb_stats_df=None,column='Total Return [%]'):
        return comb_stats_df.reset_index().sort_values(by=[column], ascending=False).set_index('index')

    # display line plot of each backtest
    st.subheader(model_name)
    total_return_plot(comb_stats_df)
    sr_df=sort_column(comb_stats_df=comb_stats_df)
    st.write(sr_df.head(n_top_results))
    st.session_state[model_name+'_results'] = sr_df.head(n_top_results)

    st.subheader("Results of Backtesting")
    tabs = st.tabs(['Result '+str(i) for i in range(1,len(df_list)+1)])
    for i in range(len(df_list)):
        with tabs[i]:
            HL_pct_plot(df_list[i])
            HL_plot(df_list[i])
            st.write(df_list[i])

    return None, None, None


def results_fun(model,save_file, win_df, loss_df):
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
    st.table(stats)
    st.subheader("Consecutive Wins Trades Table")
    st.write(win_df)
    st.subheader("Consecutive Loss Trades Table")
    st.write(loss_df)
    if save_file:
        st.success('File saved at this location. ' + os.getcwd()+'\\generated_files')


# title of Page
st.title("Step8: Evaluation by Multiple Backtest ")
st.markdown("___")

# sections
header = st.container()
filters = st.container()
results = st.container()

# try:
with header:
    # widgets to get model and data
    uploaded_models = st.file_uploader("Select Model", type=['PKL'], accept_multiple_files=True, key='m_model')
    uploaded_file = st.file_uploader("Select Data File", type=['CSV','TXT'], key='m_file')

# with st.form(key='my-form'):
with filters:
    if len(uploaded_models) and (uploaded_file is not None):
        feature_set = []
        df = pd.read_csv(uploaded_file)  
        df.columns = [i.strip() for i in df.columns]

        if 'Open time' not in df.columns.to_list():
            df['Open time'] = pd.to_datetime(df['Date'] + df['Time'])
            df.drop(columns=['Date','Time'], inplace=True)
        df.index = pd.to_datetime(df['Open time'])
        
        df.rename(columns={'Last':'Close'},inplace=True)
        df.drop(columns=['Open time'], inplace=True)         
        
        if os.path.isfile(path+"/model_log.csv"):
            log_file = pd.read_csv(path+"/model_log.csv")
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
                top_results = third_col.number_input("Select Top X Results", min_value=1, value=20)       
                
                exit_strategy = st.selectbox("Select Exit Strategy", exit_strategies)
                f1_col, f2_col, f3_col = st.columns(3)
                rr_end = 0
                rr = 0
                # Select SL or x no. of bars etc according to selected strategy
                if exit_strategy == "Fixed SL Exit":
                    # value = f2_col.number_input("Select SL Amount", min_value=1)
                    value = f1_col.number_input("Range of SL (Start)", min_value=0.1,step=0.1, value=4.0)
                    value_end = f2_col.number_input("Range of SL (End)", min_value=0.1, value=4.5, key='sl')
                    rr = f1_col.number_input("Range of Risk/Reward (Start)", min_value=0.1,step=0.1,value=1.9)
                    rr_end = f2_col.number_input("Range of Risk/Reward (End)", min_value=0.1,value=2.2, key='rr')
                    val_step = f3_col.number_input("Step Size for SL", min_value=0.1,step=0.1,value=0.1)
                    rr_step = f3_col.number_input("Step Size for Risk/Reward", min_value=0.1,value=0.1, key='s_rr')
                    # rr = f3_col.number_input("Select Risk Reward Ratio", min_value=0.1, max_value=20.0, step=0.5, value=1.9)
                elif exit_strategy == 'Last X Bars SL Exit':
                    value = f1_col.number_input("Range of Last X Number of Bars for SL (Start)", min_value=1,step=1, value=8)
                    value_end = f2_col.number_input("Range of Last X Number of Bars for SL (End)", min_value=1, value=10, key='m_sl')
                    rr = f1_col.number_input("Range of Risk/Reward (Start)", min_value=0.1,step=0.1,value=1.9)
                    rr_end = f2_col.number_input("Range of Risk/Reward (End)", min_value=0.1,value=2.2, key='rr')
                    val_step = f3_col.number_input("Step Size for X Number of Bars for SL", min_value=1,step=1,value=1)
                    rr_step = f3_col.number_input("Step Size for Risk/Reward", min_value=1,value=1, key='s_rr')
                elif exit_strategy == 'Fixed Bar Exit':
                    value = f1_col.number_input("Range of X Number of Bars (Start)", min_value=1,step=1, value=10)
                    value_end = f2_col.number_input("Range of X Number of Bars (End)", min_value=1, value=12, key='m_sl')
                    val_step = f3_col.number_input("Step Size for X Number of Bars", min_value=1,step=1,value=1)
                
                if value_end>=value and rr_end>=rr:
                    feature_set = st.multiselect("Select Input Features?", feature_set, default=default_feature_set, disabled=True)
                    save_file = f1_col.checkbox("Save File")
                
                    # Submit details and generate signals
                    if (data_start<data_end) and st.button("Submit"): 
                        # df = signal.read_file(uploaded_file)
                        tab1 = st.tabs([i.name.split('_')[0] for i in uploaded_models])
                        for idx, uploaded_model in enumerate(uploaded_models):
                            if uploaded_model.name.split('_')[0] != 'XGBoost':
                                model = joblib.load(uploaded_model)
                                # indicator_df = add_indicators(df)
                                # st.write(df)
                                # try:
                                output_df = model_pred(model, df.copy(), feature_set)
                                with tab1[idx]:
                                    if exit_strategy=='Fixed Bar Exit': 
                                        stats, win_df, loss_df = multi_backtest(df=output_df, exit_strategy=exit_strategy, value=value, value_end=value_end, 
                                                                            data_start=data_start, data_end=data_end, save_file=save_file, Fee=fee,
                                                                            n_top_results=top_results, start_time=start_time, end_time=end_time, val_step=val_step, model_name=uploaded_model.name)
                                    else:
                                        stats, win_df, loss_df = multi_backtest(df=output_df, exit_strategy=exit_strategy, value=value, value_end=value_end, 
                                                                            rr=rr, rr_end=rr_end, data_start=data_start, data_end=data_end, save_file=save_file, Fee=fee,
                                                                            n_top_results=top_results, start_time=start_time, end_time=end_time, val_step=val_step, rr_step=rr_step,
                                                                            model_name=uploaded_model.name)
                                    # st.session_state['stats_exists'] = True
                                    # st.session_state['stats_'+str(idx)] = stats
                                    # st.session_state['win_df_'+str(idx)] = win_df
                                    # st.session_state['loss_df_'+str(idx)] = loss_df
                            else:
                                pass
                            # except:
                            #     st.error("Input Features invalied or more/less than required features")
                    if data_end<=data_start:
                        st.error("Please select right part of data to train model.")
                else:
                    st.error("Your second value for SL/X number of bars/X number of bars for SL should be larger than first value.")
            else:
                st.error("Input data file not correct. Model train on these feature columns")
                st.error(default_feature_set)
                st.error("And your file have these feature columns")
                st.error(feature_set)
        else:
            st.error("Your models are not trained on same columns. So one use those column which trained on same columns.")
        

# with results:
#     if len(uploaded_models) and (uploaded_file is not None) and st.session_state['stats_exists']:
#         st.subheader("Results")

#         st.session_state['stats_exists'] = False
#         tab1, tab2, tab3 = st.tabs(['Random Forest', 'AdaBoost', 'XGBoost'])
#         # calculate statistics
#         for i,model in enumerate(uploaded_models):
#             if model.name.split('_')[0] == 'Random Forest':
#                 with tab1:
#                     stats = st.session_state['stats_'+str(i)]
#                     win_df = st.session_state['win_df_'+str(i)]
#                     loss_df = st.session_state['loss_df_'+str(i)]
#                     st.subheader(model.name)
#                     results_fun(model, save_file, win_df, loss_df)
#             elif model.name.split('_')[0] == 'AdaBoost':
#                 with tab2:
#                     stats = st.session_state['stats_'+str(i)]
#                     win_df = st.session_state['win_df_'+str(i)]
#                     loss_df = st.session_state['loss_df_'+str(i)]
#                     st.subheader(model.name)
#                     results_fun(model, save_file, win_df, loss_df)
#             elif model.name.split('_')[0] == 'XGBoost':
#                 with tab3:
#                     stats = st.session_state['stats_'+str(i)]
#                     win_df = st.session_state['win_df_'+str(i)]
#                     loss_df = st.session_state['loss_df_'+str(i)]
#                     st.subheader(model.name)
#                     results_fun(model, save_file, win_df, loss_df)
# # except:
# #     st.error("Files not Correct")