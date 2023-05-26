# Import Libraries
import streamlit as st
import pandas as pd
import altair as alt
import multiprocessing
import talib
import vectorbt as vbt
import os
from datetime import datetime

path = os.getcwd().split('Harmolight')[0]+'Harmolight\MAIN DATA SAVING'
# configure page
st.set_page_config(page_title='same',page_icon=None, layout='wide',initial_sidebar_state='auto')
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''
chk_apply_btn = False
st.markdown(css, unsafe_allow_html=True)
st.markdown('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


# ********  Reading File  *********
@st.cache(allow_output_mutation=True)
def read_file(uploaded_file):
    df = pd.read_csv(uploaded_file)    
    
    # Data Preprocessing
    df.columns=[i.strip() for i in df.columns]
    df.rename(columns={'Last':'Close'}, inplace=True)

    return df


def row_select(df,x):
    for i in range(x):
        df['Row_'+str(i)+'_Before'] = df['Close']


# ******  Plot Class Distribution Chart  *******
def ploting(df = None):
    st.write(df.head())
    st.subheader("Target Feature Class Distribution")
    classes_dist = df['Target'].value_counts()
    
    # convert series to dataframe
    classes_dist_df = pd.DataFrame({'classes':classes_dist.index, 'count':classes_dist.values})

    # add color column
    classes_dist_df['color'] = ['#6F3D86','#FFA600','#228B22'][:len(classes_dist_df)]

    # Create a chart
    chart = alt.Chart(classes_dist_df).mark_bar().encode(
        x='classes',
        y='count',
        color=alt.Color('color:N'),
        size = alt.Size(value=50)
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


# title of Page
st.title("Step4: Data Preprocessing")
st.markdown("___")

# sections
dataset = st.container()
preprocess = st.container()

# &&&&&&&&&&&  Main Coding Part  &&&&&&&&&&
# try:
with dataset:
    # upload data file
    uploaded_file = st.file_uploader("Select Data", type=['CSV','TXT'], key='preprocess')
    
    # check file upload or not
    if uploaded_file is not None:
        df = read_file(uploaded_file)
        


with preprocess:
    st.header("Preprocessing")
    st.markdown("___")
    
    if uploaded_file is not None:
        feature_set = list(set(df.columns.to_list()) - set(['Target','Open time','Date','Time']))
                
        if 'features' not in st.session_state:
            st.session_state['features'] = feature_set
            st.session_state['temp_features'] = feature_set.copy()
            st.session_state['temp_ind_features'] = []
            st.session_state['ind_df'] = ''
                
        default_feature_set = list(set(st.session_state['temp_features']) - set(['Open','High','Low','OHLC Avg']))       

        if 'feature_set1' not in st.session_state:
            st.session_state['feature_set1'] = default_feature_set

        feature_set1 = st.multiselect("Select File Features/Indicators", feature_set, default=st.session_state['feature_set1'], key='my_selectbox')
        f1_col, f2_col, _, _, _, _, _, _ = st.columns(8)
        
        if f1_col.button("Reset", key='my_reset_button'):
            st.session_state['temp_features'] = st.session_state['features'].copy()
            del st.session_state['feature_set1']
            st.experimental_rerun()

        if f2_col.button("Remove"):
            st.session_state['temp_features'] = ['HLC Avg', 'Close', 'Bid Volume', 'Ask Volume']
            del st.session_state['feature_set1']
            st.experimental_rerun()


        indicator_set = st.selectbox("Select Indicator", ['MACD','Double MACD'])
        f1_col, f2_col, f3_col = st.columns(3)
        if indicator_set == 'MACD' or indicator_set == 'Double MACD':
            fast_macd = f1_col.number_input("Fast MACD", min_value=1, value=12)
            slow_macd = f2_col.number_input("Slow MACD", min_value=1, value=26)
            signal_period = f3_col.number_input("MACD Signal Period", min_value=1, value=9)
        if indicator_set == 'Double MACD':
            fast_macd2 = f1_col.number_input("Fast MACD 2", min_value=1, value=12)
            slow_macd2 = f2_col.number_input("Slow MACD 2", min_value=1, value=26)
            signal_period2 = f3_col.number_input("MACD Signal Period 2", min_value=1, value=9)
        chk_ind = st.checkbox("Apply Indicator",value=True)

        indicator_df = pd.DataFrame({})
        if st.button("Apply"):
            chk_apply_btn = True
            if chk_ind:
                if indicator_set =='MACD':
                    for col_name in feature_set1:
                        MACD = talib.MACD(df[col_name],fastperiod=fast_macd, slowperiod=slow_macd, signalperiod=signal_period)
                        indicator_df['macd '+col_name] = MACD[0]
                        indicator_df['macd_signal '+col_name] = MACD[1]
                        indicator_df['macd_hist '+col_name] = MACD[2]

                if indicator_set =='Double MACD':
                    for col_name in feature_set1:
                        MACD = talib.MACD(df[col_name],fastperiod=fast_macd2, slowperiod=slow_macd2, signalperiod=signal_period2)
                        indicator_df['macd2 '+col_name] = MACD[0]
                        indicator_df['macd_signal2 '+col_name] = MACD[1]
                        indicator_df['macd_hist2 '+col_name] = MACD[2]
                if indicator_set =='RSI':
                    for col_name in feature_set1:
                        RSI = talib.RSI(df[col_name],windowperiod=fast_macd2)
                        indicator_df['rsi '+col_name] = RSI

                st.session_state['temp_ind_features'] = indicator_df.columns.to_list()
                st.session_state['ind_df'] = indicator_df
            
        if len(st.session_state['temp_ind_features'])>0 and chk_ind:
            ind_feature_set1 = st.multiselect("Select Indicators Feature", st.session_state['temp_ind_features'], default=st.session_state['temp_ind_features'])
            save_opt = st.radio("File Save Option", ['Save All Features', 'Save Only File Features', 'Save Only Indicator Features'])
            if st.button("Submit"):
                save_df = pd.DataFrame({})
                if save_opt == 'Save All Features':
                    all_features = feature_set1 + ind_feature_set1
                    save_df = pd.concat([df.copy(), st.session_state['ind_df'].copy()], axis=1)
                    save_df=save_df.T.drop_duplicates().T
                    print(len(all_features))
                    print(len(list(save_df.columns)))
                    print(save_df[list(set(all_features).intersection(set(save_df.columns)))])
                    # st.write(len(all_features))
                    # st.write(save_df.shape)
                    save_df = save_df[list(set(all_features).intersection(set(save_df.columns)))]
                elif save_opt == 'Save Only File Features':
                    save_df = df[feature_set1].copy()
                elif save_opt == 'Save Only Indicator Features':
                    save_df = st.session_state['ind_df'].copy()

                if 'Close' not in save_df.columns.to_list():
                    save_df['Close'] = df['Close'].copy()
                if 'Open' not in save_df.columns.to_list():
                    save_df['Open'] = df['Open'].copy() 
                if 'High' not in save_df.columns.to_list():
                    save_df['High'] = df['High'].copy() 
                if 'Low' not in save_df.columns.to_list():
                    save_df['Low'] = df['Low'].copy()       
                if 'Target' not in save_df.columns.to_list():
                    try:
                        save_df['Target'] = df['Target'].copy() 
                    except:
                        pass
                try:
                    save_df.index = df['Open time'].copy()
                except:
                    save_df['Date'] = df['Date'].copy()
                    save_df['Time'] = df['Time'].copy()
                    save_df.index = pd.to_datetime(save_df['Date']+ save_df['Time'])
                    save_df.index.name = 'Open time'
                save_df.dropna(inplace=True)
                save_df
                # name of file with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = "step4_data_{}.csv".format(timestamp)

                # save file specific folder if folder not present then first create it & then save file
                if os.path.isdir(path):
                    save_df.to_csv(path+"/"+filename)
                    st.success('File saved at this location. ' + path+'\\' + filename)
                else:
                    st.error("MAIN DATA SAVING folder not exist")
        elif chk_apply_btn:
            save_df = pd.DataFrame({})
            save_df = df[feature_set1].copy()

            if 'Close' not in save_df.columns.to_list():
                save_df['Close'] = df['Close'].copy()
            if 'Open' not in save_df.columns.to_list():
                save_df['Open'] = df['Open'].copy() 
            if 'High' not in save_df.columns.to_list():
                save_df['High'] = df['High'].copy() 
            if 'Low' not in save_df.columns.to_list():
                save_df['Low'] = df['Low'].copy()       
            if 'Target' not in save_df.columns.to_list():
                try:
                    save_df['Target'] = df['Target'].copy() 
                except:
                    pass
            try:
                save_df.index = df['Open time'].copy() 
            except:
                save_df['Date'] = df['Date'].copy()
                save_df['Time'] = df['Time'].copy()
                save_df.index.name = 'Open time'
            save_df.dropna(inplace=True)
            # name of file with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = "step4_data_{}.csv".format(timestamp)

            # save file specific folder if folder not present then first create it & then save file
            if os.path.isdir(path):
                save_df.to_csv(path+"/"+filename)
                st.success('File saved at this location. ' + path+'\\' + filename)
            else:
                st.error("MAIN DATA SAVING folder not exist")
            
# except:
#     st.error("File Incorrect. Use file generate from step2 or step3. Thank You!")