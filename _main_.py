import numpy as np
import pandas as pd
import streamlit as st

from pandas.errors import EmptyDataError
from streamlit_lottie import st_lottie
from streamlit_extras.colored_header import colored_header
from functions import is_ordinal, load_lottiefile, load_lottieurl, get_sheet_names, normalize_numpy, filter_columns

import openpyxl
import os
import datetime
import csv
import json
import requests

import linear_regression_test
import anova_test
import chi_square_test
import logistic_regression_test

from scipy.stats import chi2_contingency
from scipy.stats import chi
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

lottie_hacking = load_lottiefile("lottiefiles/hacker.json")
lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_yalmtkoy.json")

st.set_page_config(page_title="INFER-X (FN)", layout='wide', initial_sidebar_state='expanded', page_icon='👁️‍🗨️')

image,title = st.columns((1,16), gap="small")
with image:
    #st_lottie(lottie_hacking, key="hello")
    st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="high", # medium ; high
    #renderer="svg", # canvas
    height=None,
    width=None,
    key=None
)
with title:

    st.title("Inferential Statistical Tests Recommender") 

def main():
    df = pd.DataFrame()
    
    file = st.file_uploader("FYI: It is recommended to upload your data in csv format, it will help lessen the processing time.", type=["csv", "xlsx"])
    if file:
        show_sidebar = True
    else:
        show_sidebar = False
        
    if show_sidebar:
        st.sidebar.title("[STEPS]") 
        if file is not None:
            file_extension = file.name.split(".")[-1]
            
            if file_extension == "csv":
                try:
                    df = pd.read_csv(file, na_values=['N/A', 'NA', '','-'])
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col].fillna('-', inplace=True)
                        else:
                            df[col].fillna(0.0, inplace=True)
     
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df['Date'] = df['Date'].dt.date
                except pd.errors.EmptyDataError:
                    st.sidebar.warning("WARNING: The file is empty. Please select a valid file.")
                except UnboundLocalError:
                    st.sidebar.warning("WARNING: The file is not found or cannot be read. Please select a valid file.")
                            
            elif file_extension == "xlsx":
                sheet_names = get_sheet_names(file)
                if len(sheet_names) > 1:
                    selected_sheet = st.sidebar.selectbox("👉🏾 SELECT AN EXCEL SHEET:", sheet_names)
                    df = pd.read_excel(file, sheet_name=selected_sheet, na_values = ['N/A', 'NA', '','-'])
                else:
                    df = pd.read_excel(file, na_values=['N/A', 'NA', '','-'])
                
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col].fillna('-', inplace=True)
                    else:
                        df[col].fillna(0.0, inplace=True)
         
                    if df[col].dtype == 'datetime64[ns]':
                        df[col] = df[col].dt.strftime('%Y-%m-%d')

                if df.shape[0] == 0:
                    st.sidebar.warning("WARNING: The selected file or sheet is empty.")
        
            #df.replace("", 0, inplace=True)
            #st.dataframe(df)
                    
            button, xl_drop_row, xl_drop_col = st.columns((0.0001,1.5,4.5), gap="small")
            xldrop_rows = df.shape[0]
            xldrop_cols = df.shape[1]

            with xl_drop_row:
                st.markdown(f"<span style='color: blue;'>➕ Original count of rows : </span> <span style='color: black;'>{xldrop_rows}</span>", unsafe_allow_html=True)
            with xl_drop_col:
                st.markdown(f"<span style='color: blue;'>➕ Original count of columns : </span> <span style='color: black;'>{xldrop_cols}</span>", unsafe_allow_html=True)

            df_final = df
                
            levels = np.empty(df_final.shape[1], dtype=object) 
            for i, col in enumerate(df_final.columns):  
                if df_final[col].dtype == np.int64 or df_final[col].dtype == np.float64:
                    if df_final[col].dtype == np.int64:
                        levels[i] = "Discrete"
                    else:
                        levels[i] = "Continuous"
                elif df_final[col].dtype == object:
                    if df_final[col].nunique() <= 2:
                        levels[i] = "Binary"
                    else:
                        if is_ordinal(df_final[col]):
                            levels[i] = "Ordinal"
                        else:
                            levels[i] = "Nominal"

            continuous_count = 0
            discrete_count = 0
            binary_count = 0
            nominal_count = 0
            ordinal_count = 0

            for level in levels:
                if level == "Continuous":
                    continuous_count += 1
                elif level == "Discrete":
                    discrete_count += 1
                elif level == "Binary":
                    binary_count += 1
                elif level == "Nominal":
                    nominal_count += 1
                elif level == "Ordinal":
                    ordinal_count += 1

            isNumerical = []
            for columndata in df_final.columns:
                if df_final[columndata].dtype == np.int64 or df_final[columndata].dtype == np.float64:
                    isNumerical.append("Quantitative")
                else:
                    isNumerical.append("Categorical")

            recommendations = {}
            for i, level in enumerate(levels):
                if level == "Continuous":
                    recommendations[df_final.columns[i]] = ["ANOVA", "SIMPLE LINEAR REGRESSION"]
                elif level == "Binary":
                    recommendations[df_final.columns[i]] = ["ANOVA","CHI-SQUARE TEST", "LOGISTIC REGRESSION"]
                elif level == "Ordinal":
                    recommendations[df_final.columns[i]] = ["ANOVA", "CHI-SQUARE TEST"]
                elif level == "Discrete":
                    recommendations[df_final.columns[i]] = ["ANOVA","CHI-SQUARE TEST", "LOGISTIC REGRESSION"]
                elif level == "Nominal":
                    recommendations[df_final.columns[i]] = ["ANOVA","CHI-SQUARE TEST"]
                else:
                    recommendations[df_final.columns[i]] = ["NO RECOMMENDATION"]

            anova_count = 0
            single_linear_regression_count = 0
            logistic_regression_count = 0
            chi_square_count = 0

            for tests in recommendations.values():
                for test in tests:
                    if test == "ANOVA":
                        anova_count += 1
                    elif test == "SIMPLE LINEAR REGRESSION":
                        single_linear_regression_count += 1
                    elif test == "LOGISTIC REGRESSION":
                        logistic_regression_count += 1
                    elif test == "CHI-SQUARE TEST":
                        chi_square_count += 1
                        
            
            column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(df_final.columns, isNumerical)]
            column_options = [f"{col}: {level}" for col, level in zip(df_final.columns, levels)]
            
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )   
            
            st.sidebar.markdown(f"<span style='color: black;'>1️⃣ Normalize / Filter: </span> ", unsafe_allow_html=True)
            filter_checkbox = st.sidebar.checkbox("Filter Column (exclude value)",key="checkbox3")
            normalize_checkbox = st.sidebar.checkbox("Column Normalization", key="checkbox1")

            if filter_checkbox:
                df_final = filter_columns(df_final,filter_checkbox)

            if normalize_checkbox:
                numeric_cols = df._get_numeric_data().columns
                categorical_cols = df.columns.difference(numeric_cols)
                selected_cols = st.multiselect("Select which column/s to normalize:", df.columns)
                if len(selected_cols) == 0:
                    df_final = df.copy()
                else:
                    method = st.selectbox("Select sub normalization method",
                                                ("Z-Score", "Min-Max", "Decimal Scaling", "Max Absolute", "L1", "L2"))
                    numeric_selected_cols = [col for col in selected_cols if col in numeric_cols]
                    categorical_selected_cols = [col for col in selected_cols if col in categorical_cols]
                    df_norm = normalize_numpy(df_final, numeric_selected_cols, categorical_selected_cols, method)
                    not_selected_cols = df_final.columns.difference(selected_cols)
                    df_final = pd.concat([df_norm, df_final[not_selected_cols]], axis=1)
            else:
                df_final = df_final.copy()

            if normalize_checkbox and filter_checkbox:
                st.subheader("[👁️‍🗨️] Filtered & Normalized Data Preview:")
                st.dataframe(df_final)
                xldrop_rows2 = df_final.shape[0]
                st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{xldrop_rows2}</span>", unsafe_allow_html=True)
                colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )    

            elif normalize_checkbox:
                st.subheader("[👁️‍🗨️] Normalized Data Preview:")
                st.dataframe(df_final)
                xldrop_rows2 = df_final.shape[0]
                st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{xldrop_rows2}</span>", unsafe_allow_html=True)
                colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )    
            elif filter_checkbox:
                st.subheader("[👁️‍🗨️] Filtered Data Preview:")
                st.dataframe(df_final)
                xldrop_rows2 = df_final.shape[0]
                st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{xldrop_rows2}</span>", unsafe_allow_html=True)
                colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )    
                
            #st.sidebar.markdown("""---""")
            st.sidebar.write("\n")
            column = st.sidebar.selectbox("2️⃣ SELECT THE 'y' FIELD (dependent variable):", list(recommendations.keys()))
            
            
            st.sidebar.write("\n")
            test = st.sidebar.selectbox("3️⃣ SELECT AN INFERENTIAL TEST (determined by the selected 'y' field above):", recommendations[column])  
            
            if test == "SIMPLE LINEAR REGRESSION":
                linear_regression_test.linear_regression(df_final,file,column)
            elif test == "T-TEST (PAIRED)":
                anova_test.t_test_paired(df_final,file,column)
            elif test == "ANOVA":
                anova_test.anova(df_final,file,column)
                st.sidebar.info("INFO: The independent ('x') variable can be categorical (normalized), discrete or continuous  while the dependent ('y') variable can be discrete or continuous.")
            elif test == "CHI-SQUARE TEST":
                chi_square_test.chi_square(df_final,file,column)
            elif test == "LOGISTIC REGRESSION":
                logistic_regression_test.logistic_regression(df_final,file,column)
            else:
                st.write("Invalid test selected")
        
        st.sidebar.write("\n")                        
        details_checkbox = st.sidebar.checkbox("Do you want to more details?", key="checkbox0")

        if details_checkbox:
                tab1, tab2, tab3, tab4, tab5, tab6  = st.tabs(["🔽  CURRENT COLUMN TEST SUGGESTIONS  |","🔽  LEVELS OF MEASUREMENTS COUNT  |","🔽  TEST SUGGESTIONS COUNT  |", "🔽  LEVELS OF MEASUREMENTS  |", "🔽  VARIABLE COLUMN TYPE  |","🔽  ALL COLUMNSS TEST SUGGESTIONS  |"])
                if not column:
                    st.write("No column selected")
                with tab1:
                    for i, test in enumerate(recommendations[column], start=1):
                        st.write(f"{i}. <font color='blue'>{column}</font>: {test}", unsafe_allow_html=True)
                with tab4:
                    for i, option in enumerate(column_options, start=1):
                        st.write(f"{i}. <font color='blue'>{option.split(':')[0]}</font>: {option.split(':')[1]}", unsafe_allow_html=True)                    
                with tab5:
                    for i, numerical in enumerate(column_numerical, start=1):
                        st.write(f"{i}. <font color='blue'>{numerical.split(':')[0]}</font>: {numerical.split(':')[1]}", unsafe_allow_html=True)              
                with tab6:
                    for index, (column, tests) in enumerate(recommendations.items(), start=1):
                        st.write(f"{index}. <font color='blue'>{column}</font>: {', '.join(tests)}", unsafe_allow_html=True)
                        
                with tab2:
                    for i, (name, count) in enumerate(zip(["CONTINUOUS", "DISCRETE", "BINARY", "NOMINAL", "ORDINAL"], [continuous_count, discrete_count, binary_count, nominal_count, ordinal_count]), start=1):
                        st.write(f"{i}. <font color='blue'>{name}:</font> {count}", unsafe_allow_html=True)
                        
                with tab3:
                    for i, (test_name, count) in enumerate(zip(['ANOVA', 'SIMPLE LINEAR REGRESSION', 'LOGISTIC REGRESSION', 'CHI-SQUARE TEST'], [anova_count, single_linear_regression_count, logistic_regression_count, chi_square_count]), start=1):
                        st.write(f"{i}. <font color='blue'>{test_name}:</font> {count}", unsafe_allow_html=True)       

if __name__ == '__main__':
    main()
