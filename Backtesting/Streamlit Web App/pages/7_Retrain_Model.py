# streamlit libraries
from sklearn.model_selection import train_test_split
import streamlit as st
# Models Libraries
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
# from sklearn.tree import DecisionTreeClassifier
# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler
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

st.set_option('deprecation.showPyplotGlobalUse', False)

path = os.getcwd().split('Harmolight')[0]+'Harmolight\MAIN DATA SAVING'


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


def read_dataset(uploaded_file):
    df = pd.read_csv(uploaded_file)  
    df.columns = [i.strip() for i in df.columns]
    if 'Open time' not in df.columns.to_list():
        df['Open time'] = pd.to_datetime(df['Date'] + df['Time'])
        df.drop(columns=['Date','Time'], inplace=True)
    df.index = pd.to_datetime(df['Open time'])
        
    df.rename(columns={'Last':'Close'},inplace=True)
    df.drop(columns=['Open time'], inplace=True)  
    return df

def model_validation(uploaded_models, path):
    if os.path.isfile(path+"/model_log.csv"):
        log_file = pd.read_csv(path+"/model_log.csv")
        f_list = []
        models_names = [model.name for model in uploaded_models]
        filtered_df = log_file['Model Training Features'][log_file['Model Name'].isin(models_names)]
        st.write(filtered_df.duplicated(keep=False).all())
        if filtered_df.duplicated(keep=False).all() or len(filtered_df)==1:
            st.session_state['model_columns'] = filtered_df.iloc[0].split(",")
            return True
        else:
            st.error("Please select only those models which trained on same dataset. Your Selected models trained on different datasets")
            return False    
    else:
        st.error("No Model Log file Present.")
    


def dataset_validation(df):
    if 'Target' in df.columns.to_list():
        return True
    else:
        st.error("Selected datasets not have Target Column. Please use correct dataset file.")
        return False


def model_dataset_matching(df):
    if not(set(st.session_state['model_columns']) - set(df.columns.to_list())):
        return True
    else:
        st.error("Selected model/models trained on different dataset. Please use correct dataset file.")
        return False


# *********  Model Trainig Function  **********
def model_training(df=None, input_features=None, models=[], models_names=[], max_depth=3, n_estimator=100, n_core=1):
    model_scores, model_obj = {}, {}

    # apply standard scaling on dataset
    standard = StandardScaler()
    X = standard.fit_transform(df[input_features])
    Y = df['Target']

    # split dataset into train & test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    for idx, uploaded_model in enumerate(models):
        model = joblib.load(uploaded_model)
        n_estimators = model.n_estimators
        n_estimators = 200
        try:
            model.max_depth = 4
        except:
            # base_estimator
            base_estimator = model.base_estimator_
            # Set the max_depth parameter of the base estimator
            base_estimator.set_params(max_depth=4)

        # apply standard scaling on dataset
        standard = StandardScaler()
        X = standard.fit_transform(df[feature_set])
        Y = df['Target']

        # use random forest model for training
        if 'Random Forest' == models_names[idx]:
            model.fit(X_train, Y_train)
            predications = model.predict(X_test)
            model_scores['Random Forest'] = accuracy_score(predications, Y_test)
            model_obj['Random Forest'] = model
        
        # use adaboost model for training
        elif 'AdaBoost' == models_names[idx]:
            model.fit(X_train, Y_train)
            predications = model.predict(X_test)
            model_scores['AdaBoost'] = accuracy_score(predications, Y_test)
            model_obj['AdaBoost'] = model

        # use xgboost model for training
        elif 'XGBoost'  == models_names[idx]:
            Y_train1 = Y_train.copy()
            Y_test1 = Y_test.copy()
            Y_train1[Y_train1 == -1] = 2
            Y_test1[Y_test1 == -1] = 2
            model.fit(X_train, Y_train1)
            predications = model.predict(X_test)
            model_scores['XGBoost'] = accuracy_score(predications, Y_test1)
            model_obj['XGBoost'] = model
            
        # st.write(model.get_params()) 

    # results
    results = {'scores': model_scores,
               'models': model_obj}
    return results


def save_models(path=None, results={}, feature_set=[]):
    # Download Model
    st.subheader("Download Trained Model")
    model_name = list(results['models'].keys()) 
    for name in model_name:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{path}/{name}_retrained_model_{timestamp}.pkl"

        # save file specific folder if folder not present then first create it & then save file
        if os.path.isdir(path):
            with open(filename, 'wb') as f:
                joblib.dump(results['models'][name], f)

            # save log file for models
            if os.path.isfile(path+'/model_log.csv'):
                log_file = pd.read_csv(
                    path+'/model_log.csv')
                log_file.loc[len(log_file)] = [filename.split(
                    '/')[-1], (",").join(feature_set)]
                log_file.to_csv(
                    path+'/model_log.csv', index=False)
            else:
                log_file = pd.DataFrame({"Model Name": [filename.split(
                    '/')[-1]], "Model Training Features": [(",").join(feature_set)]})
                log_file.to_csv(
                    path+'/model_log.csv', index=False)
            st.success('&#10004; File saved at this location. "' + path +
                    '\\' + filename.split('/')[-1] + '"')
        else:
            st.error("MAIN DATA SAVING folder not exist")



st.title("Step7: Retrain Model")
st.markdown("___")

# widgets to get model and data
uploaded_models = st.file_uploader("Select Model", type=['PKL'], accept_multiple_files=True, key='retrain_model')
uploaded_file = st.file_uploader("Select Data File", type=['CSV','TXT'], key='retrain_data')

if len(uploaded_models) and (uploaded_file is not None):
    feature_set = []
    df = read_dataset(uploaded_file)
    if  model_validation(uploaded_models, path) and dataset_validation(df):
        if model_dataset_matching(df):
            first_col, sec_col = st.columns(2)
            default_feature_set = st.session_state['model_columns']
            max_depth = sec_col.number_input(
                        "Max Depth of Model", min_value=1, value=5)
            n_estimator = first_col.number_input(
                        "Number of Trees?", min_value=1, value=5000)
            
            feature_set = st.multiselect("Select Input Features?", default_feature_set, default=default_feature_set,disabled=True)
            save_file = st.checkbox("Save Models")

            
            models_names = [model.name.split('_')[0] for model in uploaded_models]
            # Submit details and generate signals
            if st.button("Submit"): 
                results = model_training(df, feature_set, uploaded_models, models_names,max_depth, n_estimator)
                st.success("&#10004; Training Complete")
                st.session_state['results'] = results
                st.session_state['model_trained'] = True
            
                if st.session_state['model_trained']:
                    st.subheader("Feature Importance")
                    # model_list = st.multiselect("Select Model", list(st.session_state['results']['models'].keys()))
                    tab1, tab2, tab3 = st.tabs(['Random Forest', 'AdaBoost', 'XGBoost'])
                
                    for model in models_names:
                        if model == 'Random Forest':
                            with tab1:
                                feature_importance(st.session_state['results'], feature_set, model)
                        elif model == 'AdaBoost':
                            with tab2:
                                feature_importance(st.session_state['results'], feature_set, model)
                        elif model == 'XGBoost':
                            with tab3:
                                feature_importance(st.session_state['results'], feature_set, model)
                if save_file:
                    save_models(path, results, feature_set)
