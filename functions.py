import numpy as np
import pandas as pd
import streamlit_pandas_profiling as spp
import streamlit as st
import json
import requests
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, Binarizer
from sklearn.feature_extraction import FeatureHasher
import base64
import os
import ydata_profiling as yd
from streamlit_pandas_profiling import st_profile_report
import openpyxl

#@st.experimental_memo
@st.cache_data
def is_ordinal(data):
    if pd.api.types.is_categorical_dtype(data):
        return data.cat.ordered
    else:
        return False

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

def get_sheet_names(file):
    wb = openpyxl.load_workbook(file, read_only=True)
    sheet_names = [sheet.title for sheet in wb.worksheets]
    return sheet_names

def normalize_numpy(df, numeric_cols, categorical_cols, method='Z-Score'):
    
    label_encoder = LabelEncoder()
    #df[categorical_cols] = df[categorical_cols].apply(lambda col: label_encoder.fit_transform(col) if col.dtype == 'object' else col)
    
    # Convert categorical columns to strings
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    # Apply label encoder only to categorical columns
    df[categorical_cols] = df[categorical_cols].apply(lambda col: label_encoder.fit_transform(col))

    normalization_dict = {
        'Z-Score': lambda df, numeric_cols: (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(),
        'Min-Max': lambda df, numeric_cols: (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min()),
        'Decimal Scaling': lambda df, numeric_cols: df[numeric_cols] / 10**np.ceil(np.log10(np.abs(df[numeric_cols]).max())),
        'Max Absolute': lambda df, numeric_cols: df[numeric_cols] / np.abs(df[numeric_cols]).max(),
        'L1': lambda df, numeric_cols: df[numeric_cols] / df[numeric_cols].abs().sum(),
        'L2': lambda df, numeric_cols: df[numeric_cols] / np.sqrt((df[numeric_cols] ** 2).sum())
    }

    scaler = normalization_dict[method]

    normalized_numeric = normalization_dict[method](df, numeric_cols)
    normalized_df = pd.concat([normalized_numeric, df[categorical_cols]], axis=1)
    
    return normalized_df

def filter_columns(df_final, filter_checkbox):
    
    if filter_checkbox:

        columns = df_final.columns.tolist()
        selected_columns = st.sidebar.multiselect("👉 COLUMN: Select which column/s to filter:", columns)
        if selected_columns:
            datetime_columns = df_final.select_dtypes(include=['datetime64']).columns.tolist()
            for selected_column in selected_columns:
                if selected_column in datetime_columns:
                    df_final[selected_column] = df_final[selected_column].dt.date.strftime('%Y-%m-%d')
                filter_values = sorted(df_final[selected_column].unique().tolist())
                filter_values = st.sidebar.multiselect("👉 EXCLUDE: Select filter values for {}".format(selected_column), filter_values)
                if filter_values:
                    df_final = df_final[~df_final[selected_column].isin(filter_values)]
    
    return df_final

def transform_column(df_final, col_name, method='Ordinal'):

    encoding_dict = {
        'Label': lambda df_final, col_name: LabelEncoder().fit_transform(df_final[col_name]),
        'One-Hot': lambda df_final, col_name: OneHotEncoder(sparse=False, handle_unknown='ignore').fit_transform(df_final[[col_name]]),
        'Binary': lambda df_final, col_name: Binarizer(threshold=0.0).fit_transform(df_final[[col_name]]),
        'Ordinal': lambda df_final, col_name: OrdinalEncoder().fit_transform(df_final[[col_name]]),
        'Count': lambda df_final, col_name: df_final[col_name].map(df_final[col_name].value_counts()),
        'Frequency': lambda df_final, col_name: df_final[col_name].map(df_final[col_name].value_counts()).fillna(0)
    }
    encoder = encoding_dict[method]

    if method == 'Binary' and df_final[col_name].dtype != 'object':
     
        df_final = encoder(df_final, col_name)
        df_final.columns = [f'{col_name}_binarized']
    else:
        df_final[col_name] = encoder(df_final, col_name)

    return df_final[col_name]

def download_csv(file_name):
    # Download the data as a CSV file
    with open(file_name, 'rb') as f:
            data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            href = f'💾 <a href="data:file/html;base64,{b64}" download="{file_name}" target="_blank">Download report</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)   
    return file_name
    
def remove_file(file_name):
    os.remove(file_name)  


def transformation_check(df_final, isNumerical, column, test):

    if isNumerical[df_final.columns.get_loc(column)] == 'Categorical' and (test == 'CHI-SQUARE' or test == 'ANOVA'):
       
        numerical_columns = df_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
        levels = ['Discrete' if pd.api.types.is_integer_dtype(df_final[col]) else 'Continuous' for col in numerical_columns]
        levels += ['Binary' if df_final[col].nunique() == 2 else 'Ordinal' if is_ordinal(df_final[col]) else 'Nominal' for col in df_final.columns if col not in numerical_columns]

        column_index = df_final.columns.get_loc(column)
        if levels[column_index] == "Nominal" and len(df_final[column].unique()) > 2:
            recommended_method = "One-Hot"
        elif levels[column_index] == "Binary" and len(df_final[column].unique()) == 2:
            recommended_method = "Label"
        elif levels[column_index] == "Ordinal":
            recommended_method = "Ordinal"
        else:
            recommended_method = None

        if recommended_method:
            method = recommended_method
        else:
            method = "Label"

        transformed_col = transform_column(df_final, column, method)

        df_final[column] = transformed_col

        return df_final
