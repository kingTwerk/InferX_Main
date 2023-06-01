import numpy as np
import pandas as pd
import ydata_profiling as yd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

from pandas.errors import EmptyDataError
from streamlit_lottie import st_lottie
from streamlit_extras.colored_header import colored_header
from functions import transformation_check, is_ordinal, load_lottiefile, load_lottieurl, get_sheet_names, filter_columns, remove_file, download_csv
from collections import defaultdict, Counter

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

import base64
import plotly.express as px
import webbrowser
import time

st.set_page_config(page_title="INFER-X (Local-updated 060123)", layout='wide', initial_sidebar_state='expanded', page_icon="👁️‍🗨️")


lottie_hacking = load_lottiefile("lottiefiles/hacker.json")
lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_yalmtkoy.json")

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.title("🕵🏽 Inferential Statistical Tests Recommender (INFER-X)") 
# st.markdown(""" <style> .font {                                          
# font-size:30px ; font-family: 'Cooper Black'; color: #FF9620;} 
# </style> """, unsafe_allow_html=True)
# st.markdown('<p class="font">Inferential Statistical Tests Recommender</p>', unsafe_allow_html=True)
          
def main():
    padding = 0
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)
    
    df = pd.DataFrame()

    file = st.file_uploader("FYI: It is recommended to upload your data in csv format, it will help lessen the processing time.", type=["csv", "xlsx"])

    if file:
        show_sidebar = True
    else:
        show_sidebar = False

    if show_sidebar: 
        if file is not None:
            file_extension = file.name.split(".")[-1]
            na_values = ['N/A', 'NA', ' N/A', ' NA', 'N/A ', 'NA ', ' N/A ', ' NA ', '', '-', ' - ', ' -', '- ']
            if file_extension == "csv":
                try:
                    df = pd.read_csv(file, na_values=na_values)
                    df = df.apply(lambda x: x.str.strip() if isinstance(x[0], str) else x)           
                except pd.errors.EmptyDataError:
                    st.sidebar.warning("WARNING: The file is empty. Please select a valid file.")
                except UnboundLocalError:
                    st.sidebar.warning("WARNING: The file is not found or cannot be read. Please select a valid file.")

            elif file_extension == "xlsx":
                sheet_names = get_sheet_names(file)
                if len(sheet_names) > 1:
                    selected_sheet = st.sidebar.selectbox("👉 SELECT AN EXCEL SHEET:", sheet_names)
                    df = pd.read_excel(file, sheet_name=selected_sheet, na_values=na_values)
                else:
                    df = pd.read_excel(file, na_values=na_values)
                df = df.apply(lambda x: x.str.strip() if isinstance(x[0], str) else x)
                if df.shape[0] == 0:
                    st.sidebar.warning("WARNING: The selected file or sheet is empty.")
            
            button, xl_drop_row, xl_drop_col = st.columns((0.0001,1.5,4.5), gap="small")
            
            xldrop_rows = df.shape[0]
            xldrop_cols = df.shape[1]

            with xl_drop_row:
                st.markdown(f"<span style='color: blue;'>➕ Original # of rows : </span> <span style='color: black;'>{xldrop_rows}</span>", unsafe_allow_html=True)
            with xl_drop_col:
                st.markdown(f"<span style='color: blue;'>➕ Original # of columns : </span> <span style='color: black;'>{xldrop_cols}</span>", unsafe_allow_html=True)

            st.sidebar.title("👟 Infer-X Steps:") 

            object_cols = df.select_dtypes(include=['object']).columns
            int_cols = df.select_dtypes(include=['int64']).columns
            float_cols = df.select_dtypes(include=['float64']).columns
            date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

            df[object_cols] = df[object_cols].fillna('-')
            df[int_cols] = df[int_cols].fillna(0)
            df[float_cols] = df[float_cols].fillna(0.0)

            for col in date_cols:
                df[col] = df[col].dt.strftime('%Y-%m-%d')
  
            df_final = df
                
            levels = np.empty(df_final.shape[1], dtype=object) 
            for i, col in enumerate(df_final.columns):  
                
                if df_final[col].dtype == np.int64:
                    if df_final[col].nunique() == 2:
                        levels[i] = "Binary"
                    else:
                        levels[i] = "Discrete"
                        
                elif df_final[col].dtype == np.float64:
                    if df_final[col].nunique() == 2:
                        levels[i] = "Binary"
                    else:
                        levels[i] = "Continuous"
                        
                elif df_final[col].dtype == object:
                    if df_final[col].nunique() == 2:
                        levels[i] = "Binary"
                    else:
                        if is_ordinal(df_final[col]):
                            levels[i] = "Ordinal"
                        else:
                             levels[i] = "Nominal"

                count_dict = defaultdict(int)

                for level in levels:
                    count_dict[level] += 1

                continuous_count = count_dict["Continuous"]
                discrete_count = count_dict["Discrete"]
                binary_count = count_dict["Binary"]
                nominal_count = count_dict["Nominal"]
                ordinal_count = count_dict["Ordinal"]

            isNumerical = ["Quantitative" if pd.api.types.is_numeric_dtype(df_final[columndata]) else "Categorical" for columndata in df_final.columns]

            level_to_recommendation = {
                "Continuous": ["ANOVA", "SIMPLE LINEAR REGRESSION"],
                "Discrete": ["ANOVA"],
                "Binary": ["LOGISTIC REGRESSION", "CHI-SQUARE"],
                "Ordinal": ["CHI-SQUARE"],
                "Nominal": ["CHI-SQUARE"],
            }

            recommendations = {col: level_to_recommendation[level] for col, level in zip(df_final.columns, levels)}

            test_to_count = {
                "ANOVA": 0,
                "SIMPLE LINEAR REGRESSION": 0,
                "LOGISTIC REGRESSION": 0,
                "CHI-SQUARE": 0,
            }

            for tests in recommendations.values():
                for test in tests:
                    test_to_count[test] += 1

            anova_count = test_to_count["ANOVA"]
            single_linear_regression_count = test_to_count["SIMPLE LINEAR REGRESSION"]
            logistic_regression_count = test_to_count["LOGISTIC REGRESSION"]
            chi_square_count = test_to_count["CHI-SQUARE"]
            
            column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(df_final.columns, isNumerical)]
            column_options = [f"{col}: {level}" for col, level in zip(df_final.columns, levels)]
            
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )   
            
            st.sidebar.write("1️⃣ FILTER BY COLUMN:")
            filter_checkbox = st.sidebar.checkbox("Filter (to exclude values)",key="checkbox3")
            
            if filter_checkbox: 
                if filter_checkbox:
                    df_final = filter_columns(df_final, filter_checkbox)
            else:
                df_final = df.copy()   
                 
            column = st.sidebar.selectbox("2️⃣ SELECT THE 'y' FIELD (dependent variable):", list(recommendations.keys()))            
            test = st.sidebar.selectbox("3️⃣ SELECT AN INFERENTIAL TEST (determined by the selected 'y' field above):", recommendations[column])  

            transformation_check(df_final, isNumerical, column, test)
            
            if test == "SIMPLE LINEAR REGRESSION":
                linear_regression_test.linear_regression(df_final,file,column)
            elif test == "T-TEST (PAIRED)":
                anova_test.t_test_paired(df_final,file,column)
            elif test == "ANOVA":
                anova_test.anova(df_final,file,column)
            elif test == "CHI-SQUARE":
                chi_square_test.chi_square(df_final,file,column,test)
            elif test == "LOGISTIC REGRESSION":
                logistic_regression_test.logistic_regression(df_final,file,column)
            else:
                st.write("Invalid test selected")
        
        st.sidebar.write("\n")                        
        details_checkbox = st.sidebar.checkbox("Do you want to more details?", key="checkbox0")

        if details_checkbox:
                colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )   
                st.subheader("[🔍] Variable Insights:")
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Ⅰ.┊NEEDS NORMALIZATION┊","Ⅱ.┊MEASUREMENT TYPES┊","Ⅲ.┊TEST SUGGESTIONS┊", "Ⅳ.┊LEVELS OF MEASUREMENTS┊", "Ⅴ.┊VARIABLE TYPES┊","Ⅵ.┊ALL TEST SUGGESTIONS┊","Ⅶ.┊UNIQUE VARIABLE┊","Ⅷ.┊PANDAS DTYPE┊"])
                if not column:
                    st.write("No column selected")
                with tab8:
                    for i, (col_name, dtype) in enumerate(df_final.dtypes.iteritems(), start=1):
                        st.write(f"{i}.&nbsp; <font color='blue'>{col_name}</font>: {dtype}", unsafe_allow_html=True)
                    
                with tab7:
                    for i, option in enumerate(column_options, start=1):
                        name, dtype = option.split(":")
                        st.write(f"{i}.&nbsp; <font color='blue'>{name.strip()} ({dtype.strip()}):</font> {df_final[name.strip()].nunique()} unique values", unsafe_allow_html=True)
                with tab2:
                    for i, (name, dtype, count) in enumerate(zip(["CONTINUOUS", "DISCRETE", "BINARY", "NOMINAL", "ORDINAL"], ["float64", "int64", "int64", "object", "object"], [continuous_count, discrete_count, binary_count, nominal_count, ordinal_count]), start=1):
                        st.write(f"{i}.&nbsp; <font color='blue'>{name} ({dtype}):</font> {count}", unsafe_allow_html=True)                     
                with tab3:
                    for i, (test_name, count) in enumerate(zip(['ANOVA', 'SIMPLE LINEAR REGRESSION', 'LOGISTIC REGRESSION', 'CHI-SQUARE'], [anova_count, single_linear_regression_count, logistic_regression_count, chi_square_count]), start=1):
                        st.write(f"{i}.&nbsp; <font color='blue'>{test_name} :</font> {count}", unsafe_allow_html=True)  
                with tab4:  
                    for i, option in enumerate(column_options, start=1):
                        st.write(f"{i}.&nbsp; <font color='blue'>{option.split(':')[0]} </font>: {option.split(':')[1]}", unsafe_allow_html=True)      
                with tab5:
                    for i, numerical in enumerate(column_numerical, start=1):
                        st.write(f"{i}.&nbsp; <font color='blue'>{numerical.split(':')[0]}</font>: {numerical.split(':')[1]}", unsafe_allow_html=True)              
                with tab6:
                    for index, (column, tests) in enumerate(recommendations.items(), start=1):
                        st.write(f"{index}.&nbsp; <font color='blue'>{column}</font>: {', '.join(tests)}", unsafe_allow_html=True)       
                with tab1:
                    # Check if each column needs normalization using z-score method
                    needsNormalization = []
                    for column in df.columns:
                        if df[column].dtype == np.int64 or df[column].dtype == np.float64:
                            z_scores = (df[column] - df[column].mean()) / df[column].std()
                            if (z_scores.max() - z_scores.min()) > 3: # Check if the range of z-scores is greater than 3
                                needsNormalization.append((column, z_scores))

                    # List the columns that need normalization
                    if len(needsNormalization) > 0:
                        for i, (column, z_scores) in enumerate(needsNormalization, start=1):
                            st.write(f"{i}.&nbsp; <font color='blue'>{column}</font> (z-score: {z_scores.max() - z_scores.min():.2f})", unsafe_allow_html=True)
                    else:
                        st.write("No columns need to be normalized.")
                                
        if st.sidebar.checkbox("Generate an extensive report providing insights into various aspects of your data.", key="pandas_profiling"): 

            option1=st.sidebar.radio(
            'What variables do you want to include in the report?',
            ('All variables', 'A subset of variables'))
            
            if option1=='All variables':
                df3=df
            
            elif option1=='A subset of variables':
                var_list=list(df.columns)
                option3=st.sidebar.multiselect(
                    'Select variable(s) you want to include in the report.',
                    var_list)
                df3=df[option3]
        
            option2 = st.sidebar.selectbox(
            'Choose Minimal Mode or Complete Mode',
            ('Minimal Mode', 'Complete Mode'))

            if option2=='Complete Mode':
                mode='complete'
                st.sidebar.warning('FYI: May cause the app to run overtime or fail for large datasets due to computational limit.')
            elif option2=='Minimal Mode':
                mode='minimal'
                st.sidebar.warning('FYI: Disables expensive computations such as correlations and duplicate row detection.')

            if st.sidebar.button('Generate Report'):
                if mode=='complete':
                    profile=yd.ProfileReport(df3,
                        minimal=False,
                        title="User uploaded table",
                        progress_bar=True,
                    )
                    colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )  

                    st_profile_report(profile) 
                    file_name = f"raw_data_overview_{file.name.split('.')[0]}.html"
                    profile.to_file(file_name)

                    download_csv(file_name)        
                    remove_file(file_name)
                            
                elif mode=='minimal':
                    profile=yd.ProfileReport(df3,
                        minimal=True,
                        title="User uploaded table",
                        progress_bar=True,
                    )
                    colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )  
                    st.subheader("[🖨️] RAW Data Report:")
                    st_profile_report(profile) 
                    file_name = f"raw_data_overview_{file.name.split('.')[0]}.html"
                    profile.to_file(file_name)

                    download_csv(file_name)        
                    remove_file(file_name)

if __name__ == '__main__':
    main()
