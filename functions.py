import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.colored_header import colored_header
import json
import requests
from sklearn.preprocessing import LabelEncoder

#@st.experimental_memo
@st.cache_data
def is_ordinal(data):
    try:
        cat_data = pd.Categorical(data)
        return cat_data.ordered
    except ValueError:
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
    xl = pd.read_excel(file, sheet_name=None)
    sheet_names = list(xl.keys())
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
        selected_columns = st.multiselect("Select which column/s to filter:", columns)
        if selected_columns:
            for selected_column in selected_columns:
                if df_final[selected_column].dtype == 'datetime64[ns]':
                    df_final[selected_column] = df_final[selected_column].dt.strftime('%Y-%m-%d')
                filter_values = sorted(df_final[selected_column].unique().tolist())
                filter_values = st.multiselect("EXCLUDE: Select filter values for {}".format(selected_column), filter_values)
                if filter_values:
                    df_final = df_final[~df_final[selected_column].isin(filter_values)]
    
    return df_final

def transform_column(df_final, col_name, method='Ordinal'):
    # Define dictionary of encoding methods
    encoding_dict = {
        'Label': lambda df_final, col_name: LabelEncoder().fit_transform(df_final[col_name]),
        'One-Hot': lambda df_final, col_name: OneHotEncoder(sparse=False, handle_unknown='ignore').fit_transform(df_final[[col_name]]),
        'Binary': lambda df_final, col_name: Binarizer(threshold=0.0).fit_transform(df_final[[col_name]]),
        'Ordinal': lambda df_final, col_name: OrdinalEncoder().fit_transform(df_final[[col_name]]),
        #'Count': lambda df_final, col_name: df_final.groupby(col_name).size().reset_index(name=f'{col_name}_count')[f'{col_name}_count'],
        #'Count': lambda df_final, col_name: df_final.groupby(col_name).size().reset_index(name=f'{col_name}_count'),
        'Count': lambda df_final, col_name: df_final[col_name].map(df_final[col_name].value_counts()),
        #'Hashing': lambda df_final, col_name: FeatureHasher(n_features=10, input_type='string').transform(df_final[[col_name]].astype(str)).toarray(),
        #'Hashing': lambda df_final, col_name: ce.HashingEncoder(n_components=10).fit_transform(df_final[col_name]),
    
        'Frequency': lambda df_final, col_name: df_final[col_name].map(df_final[col_name].value_counts()).fillna(0)
    }
    # Select encoding method
    encoder = encoding_dict[method]
    # Apply encoding to column
    if method == 'Binary' and df_final[col_name].dtype != 'object':
        # Only apply binary encoding to numeric columns
        df_final = encoder(df_final, col_name)
        df_final.columns = [f'{col_name}_binarized']
    else:
        df_final[col_name] = encoder(df_final, col_name)
    # Return transformed column
    return df_final[col_name]
