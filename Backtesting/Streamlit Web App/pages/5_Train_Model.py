# ***** Import Libraries *****
# streamlit libraries
import streamlit as st
# Models Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# Model Storage
import joblib
import multiprocessing
import os
from datetime import datetime
# Visualization libraries
import matplotlib.pyplot as plt
import altair as alt

print(os.getcwd())
path = os.getcwd()+'\generated_files'

# Page configuration settings
st.set_page_config(page_title='same', page_icon=None,
                   layout='wide', initial_sidebar_state='auto')
st.set_option('deprecation.showPyplotGlobalUse', False)

# title of Page
st.title("Step5: Model Training")
st.markdown("___")
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''
st.markdown(css, unsafe_allow_html=True)
if 'data_load' not in st.session_state:
    st.session_state['data_load'] = False
    st.session_state['model_trained'] = False
    st.session_state['results'] = None


# ********  Function to visualize important models feature columns  ********
def feature_importance(results, feature_set, name):
    st.subheader(str(name)+" Model")
    st.write("Score : "+str(results['scores'][name]))

    # Get the feature importance scores
    importances = results['models'][name].feature_importances_

    # Sort the feature importances in descending order
    sorted_idx = importances.argsort()[::-1]
    x = np.array(feature_set)[sorted_idx]

    # Plot the feature importances
    plt.bar(range(len(x[:20])), importances[sorted_idx[:20]])
    plt.xticks(range(len(x[:20])), feature_set[:20], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importances - '+str(name))
    st.pyplot()


# ********  Reading File  *********
@st.cache(allow_output_mutation=True)
def read_file(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # #### Data Preprocessing ####
    df.columns = [i.strip() for i in df.columns]

    return df


# ******  Plot Class Distribution Chart  *******
def ploting(df=None):
    st.write(df.head())
    st.subheader("Target Feature Class Distribution")
    classes_dist = df['Target'].value_counts()

    # convert series to dataframe
    classes_dist_df = pd.DataFrame(
        {'classes': classes_dist.index, 'count': classes_dist.values})

    # add color column
    classes_dist_df['color'] = ['#6F3D86',
                                '#FFA600', '#228B22'][:len(classes_dist_df)]

    # Create a chart
    chart = alt.Chart(classes_dist_df).mark_bar().encode(
        x='classes',
        y='count',
        color=alt.Color('color:N'),
        size=alt.Size(value=50)
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


# *********  Model Trainig Function  **********
def model_training(df=None, input_features=None, models=[], max_depth=3, n_estimator=100, n_core=1, smote_check=False):
    model_scores, model_obj = {}, {}

    # apply standard scaling on dataset
    standard = StandardScaler()
    X = standard.fit_transform(df[input_features])
    Y = df['Target']

    try:
        # apply smote technique
        if smote_check:
            # use SMOTE to oversample the minority class
            oversample = SMOTE()
            X, Y = oversample.fit_resample(X, Y)
    except:
        st.warning("You can not train with smote. To apply smote technique your Target column should have more than 6 entry signals.")

    # split dataset into train & test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    # use random forest model for training
    if 'Random Forest' in models:
        SRF = RandomForestClassifier(
            n_estimators=n_estimator, max_depth=max_depth, random_state=0, n_jobs=n_core)
        SRF.fit(X_train, Y_train)
        predications = SRF.predict(X_test)
        model_scores['Random Forest'] = accuracy_score(predications, Y_test)
        model_obj['Random Forest'] = SRF

    # use adaboost model for training
    if 'AdaBoost' in models:
        base_model = DecisionTreeClassifier(max_depth=2)
        ada_model = AdaBoostClassifier(
            base_estimator=base_model, n_estimators=n_estimator)
        ada_model.fit(X_train, Y_train)
        predications = ada_model.predict(X_test)
        model_scores['AdaBoost'] = accuracy_score(predications, Y_test)
        model_obj['AdaBoost'] = ada_model

    # use xgboost model for training
    if 'XGBoost' in models:
        Y_train1 = Y_train.copy()
        Y_test1 = Y_test.copy()
        Y_train1[Y_train1 == -1] = 2
        Y_test1[Y_test1 == -1] = 2
        xg_reg = XGBClassifier(eval_metric='mlogloss',
                               n_estimators=n_estimator)
        xg_reg.fit(X_train, Y_train1)
        predications = xg_reg.predict(X_test)
        model_scores['XGBoost'] = accuracy_score(predications, Y_test1)
        model_obj['XGBoost'] = xg_reg

    # results
    results = {'scores': model_scores,
               'models': model_obj}
    return results


# sections
header = st.container()
dataset = st.container()
features = st.container()
model_train = st.container()

# &&&&&&&&&&&  Main Coding Part  &&&&&&&&&&
# try:
with dataset:
    st.header("Dataset Selection")

    # upload data file
    uploaded_file = st.file_uploader("Select Data", type=['CSV', 'TXT'])

    # check file upload or not
    if uploaded_file is not None:
        df = read_file(uploaded_file)
        if 'Target' in df.columns.to_list():
            ploting(df)
            st.session_state['data_load'] = True
        else:
            st.error(
                "This data not have Target column. Please use correct data file.")
            st.session_state['data_load'] = False

with model_train:
    if (uploaded_file is not None) and st.session_state['data_load']:
        st.header("Model Training")
        st.text(
            "We will use different models like Random Forest, XGBoost & etc to train model.")

        # models list
        feature_set = []
        model_set = ['Random Forest', 'XGBoost', 'AdaBoost']

        # model training parameters
        with st.form(key="my-form"):
            try:
                feature_set = list(df.columns.drop(['Target', 'Open time']))
            except:
                feature_set = list(df.columns.drop(['Target', 'Date', 'Time']))
            first_col, sec_col, third_col = st.columns(3)

            # Model parameters
            data_start = first_col.number_input(
                "Select Starting Data for Training (%)", min_value=0, max_value=100, value=0)
            data_end = sec_col.number_input(
                "Select Ending Data for Training (%)", min_value=1, max_value=100, value=100)
            max_depth = third_col.number_input(
                "Max Depth of Model", min_value=1, value=5)
            n_cores = first_col.number_input(
                "Number of CPU Cores", min_value=1, max_value=multiprocessing.cpu_count(), step=1, value=8)
            n_estimator = sec_col.number_input(
                "Number of Trees?", min_value=1, value=5000)
            model_set = third_col.multiselect(
                "Select Model", model_set, default=model_set)
            smote_check = st.checkbox("Apply SMOTE")
            feature_set = st.multiselect(
                "Select Input Features?", feature_set, default=feature_set)

            # Model Scores Section
            if st.form_submit_button("Submit"):
                if len(feature_set) and (data_start < data_end):
                    # model training
                    df = df.iloc[int((data_start/100)*len(df)):int((data_end/100)*len(df))]
                    results = model_training(
                        df, feature_set, model_set, max_depth, n_estimator, n_cores, smote_check)
                    st.success("&#10004; Training Complete")
                    st.session_state['results'] = results
                    st.session_state['model_trained'] = True
                    
                    # Model Results
                    st.subheader("Model Scores")
                    for key in results['scores'].keys():
                        st.write(f'{key} : {results["scores"][key]}')
                    
                    # Download Model
                    st.subheader("Download Trained Model")
                    model_name = list(results['models'].keys()) 
                    for name in model_name:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{path}/{name}_model_{timestamp}.pkl"

                        # save file specific folder if folder not present then first create it & then save file
                        if os.path.isdir(path):
                            with open(filename, 'wb') as f:
                                joblib.dump(results['models'][name], f)

                            # save log file for models
                            if os.path.isfile('model_log.csv'):
                                log_file = pd.read_csv('model_log.csv')
                                log_file.loc[len(log_file)] = [filename.split(
                                    '/')[-1], (",").join(feature_set)]
                                log_file.to_csv('model_log.csv', index=False)
                            else:
                                log_file = pd.DataFrame({"Model Name": [filename.split(
                                    '/')[-1]], "Model Training Features": [(",").join(feature_set)]})
                                log_file.to_csv('model_log.csv', index=False)
                            st.success('&#10004; File saved at this location. "' + path +
                                    '\\' + filename.split('/')[-1] + '"')
                        else:
                            st.error("MAIN DATA SAVING folder not exist")

                else:
                    st.error("""Please select atleast one or more column/feature to train model. 
                                Or your starting training data value more than ending training data value.
                                please use smaller value than ending training data value. Thank You!""")

        if st.session_state['model_trained']:
            st.subheader("Feature Importance")
            # model_list = st.multiselect("Select Model", list(st.session_state['results']['models'].keys()))
            tab1, tab2, tab3 = st.tabs(['Random Forest', 'AdaBoost', 'XGBoost'])
        
            for model in model_set:
                if model == 'Random Forest':
                    with tab1:
                        feature_importance(st.session_state['results'], feature_set, model)
                elif model == 'AdaBoost':
                    with tab2:
                        feature_importance(st.session_state['results'], feature_set, model)
                elif model == 'XGBoost':
                    with tab3:
                        feature_importance(st.session_state['results'], feature_set, model)
                
# except:
#     st.write("Dataset not Correct")
