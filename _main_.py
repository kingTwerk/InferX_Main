import numpy as np
import pandas as pd
import ydata_profiling as yd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

from pandas.errors import EmptyDataError
from streamlit_lottie import st_lottie
from streamlit_extras.colored_header import colored_header
from functions import transformation_check, is_ordinal, load_lottiefile, load_lottieurl, get_sheet_names, normalize_numpy, filter_columns, transform_column, remove_file, download_csv

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

st.set_page_config(page_title="INFER-X (051023)", layout='wide', initial_sidebar_state='expanded', page_icon="👁️‍🗨️")


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

            if file_extension == "csv":
                try:
                    df = pd.read_csv(file, na_values=['N/A', 'NA',' N/A', ' NA','N/A ', 'NA ',' N/A ', ' NA ', '','-',' - ',' -', '- '])
                    df = df.apply(lambda x: x.str.strip() if isinstance(x[0], str) else x)           
                except pd.errors.EmptyDataError:
                    st.sidebar.warning("WARNING: The file is empty. Please select a valid file.")
                except UnboundLocalError:
                    st.sidebar.warning("WARNING: The file is not found or cannot be read. Please select a valid file.")

            elif file_extension == "xlsx":
                sheet_names = get_sheet_names(file)
                if len(sheet_names) > 1:
                    selected_sheet = st.sidebar.selectbox("👉 SELECT AN EXCEL SHEET:", sheet_names)
                    df = pd.read_excel(file, sheet_name=selected_sheet, na_values = ['N/A', 'NA',' N/A', ' NA','N/A ', 'NA ',' N/A ', ' NA ', '','-',' - ',' -', '- '])
                else:
                    df = pd.read_excel(file, na_values= ['N/A', 'NA',' N/A', ' NA','N/A ', 'NA ',' N/A ', ' NA ', '','-',' - ',' -', '- '])
                df = df.apply(lambda x: x.str.strip() if isinstance(x[0], str) else x)
                if df.shape[0] == 0:
                    st.sidebar.warning("WARNING: The selected file or sheet is empty.")
            
            # create a list of column options
            # column_options = [f"{col}: {df[col].dtype}" for col in df.columns]

            # loop over the column options and display information for each column
            # for i, option in enumerate(column_options, start=1):
            #     name, dtype = option.split(":")
            #     num_blank = df[name.strip()].isna().sum()  # count the number of blank records
            #     if num_blank > 0:  # display the information only if there is at least one blank record
            #         st.write(f"{i}.&nbsp; <font color='blue'>{name.strip()} ({dtype.strip()}):</font> {num_blank} blank records", unsafe_allow_html=True)

            button, xl_drop_row, xl_drop_col = st.columns((0.0001,1.5,4.5), gap="small")
            
            xldrop_rows = df.shape[0]
            xldrop_cols = df.shape[1]

            with xl_drop_row:
                st.markdown(f"<span style='color: blue;'>➕ Original # of rows : </span> <span style='color: black;'>{xldrop_rows}</span>", unsafe_allow_html=True)
            with xl_drop_col:
                st.markdown(f"<span style='color: blue;'>➕ Original # of columns : </span> <span style='color: black;'>{xldrop_cols}</span>", unsafe_allow_html=True)

            st.sidebar.title("👟 Infer-X Steps:") 
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna('-', inplace=True)
                elif df[col].dtype == 'int64':
                    df[col].fillna(0, inplace=True)
                elif df[col].dtype == 'float64':
                    df[col].fillna(0.0, inplace=True)
                    
                if df[col].dtype == 'datetime64[ns]':
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
                    
            df_final = df
                
            # levels = np.empty(df_final.shape[1], dtype=object) 
            # for i, col in enumerate(df_final.columns):  
            #     if df_final[col].dtype == np.int64 or df_final[col].dtype == np.float64:
            #         if df_final[col].dtype == np.int64:
            #             levels[i] = "Discrete"
            #         else:
            #             levels[i] = "Continuous"
            #     elif df_final[col].dtype == object:
            #         if df_final[col].nunique() <= 2:
            #             levels[i] = "Binary"
            #         else:
            #             if is_ordinal(df_final[col]):
            #                 levels[i] = "Ordinal"
            #             else:
            #                 levels[i] = "Nominal"
            
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
            
            
            continuous_count, discrete_count, binary_count, nominal_count, ordinal_count = 0, 0, 0, 0, 0

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
                elif level == "Discrete":
                    recommendations[df_final.columns[i]] = ["ANOVA"]                    
                elif level == "Binary":
                    recommendations[df_final.columns[i]] = ["LOGISTIC REGRESSION"]
                elif level == "Ordinal":
                    recommendations[df_final.columns[i]] = ["CHI-SQUARE"]
                elif level == "Nominal":
                    recommendations[df_final.columns[i]] = ["CHI-SQUARE"]
                else:
                    recommendations[df_final.columns[i]] = ["NO RECOMMENDATION"]

            anova_count, single_linear_regression_count, logistic_regression_count, chi_square_count = 0, 0, 0, 0

            for tests in recommendations.values():
                for test in tests:
                    if test == "ANOVA":
                        anova_count += 1
                    elif test == "SIMPLE LINEAR REGRESSION":
                        single_linear_regression_count += 1
                    elif test == "LOGISTIC REGRESSION":
                        logistic_regression_count += 1
                    elif test == "CHI-SQUARE":
                        chi_square_count += 1
                               
            column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(df_final.columns, isNumerical)]
            column_options = [f"{col}: {level}" for col, level in zip(df_final.columns, levels)]
            
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )   
            
            # st.sidebar.write("1️⃣ Filter / Normalize / Transform Columns :")
            st.sidebar.write("1️⃣ FILTER BY COLUMN:")
            filter_checkbox = st.sidebar.checkbox("Filter (to exclude values)",key="checkbox3")
            
            if filter_checkbox: # or transform_checkbox:
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
        #profiling_checkbox = st.sidebar.checkbox("Raw file profiling?", key="checkbox10")

        if details_checkbox:
                colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )   
                # tab1, tab2, tab3, tab4, tab5, tab6, tab7  = st.tabs(["1️⃣ CURRENT COLUMN TEST SUGGESTIONS  |","2️⃣ LEVELS OF MEASUREMENTS COUNT  |","3️⃣ TEST SUGGESTIONS COUNT  |", "4️⃣ LEVELS OF MEASUREMENTS  |", "5️⃣ VARIABLE COLUMN TYPE  |","6️⃣ ALL COLUMNSS TEST SUGGESTIONS  |","7️⃣ NEEDS NORMALIZATION  |"])
                # tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["𝟏┊NEEDS NORMALIZATION┊","𝟐┊MEASUREMENT TYPES┊","𝟑┊TEST SUGGESTIONS┊", "𝟒┊LEVELS OF MEASUREMENTS┊", "𝟓┊VARIABLE TYPES┊","𝟔┊ALL TEST SUGGESTIONS┊","Ⅰ. VARIABLE UNIQUE COUNT┊"])
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
                    # for i, (name, count) in enumerate(zip(["CONTINUOUS", "DISCRETE", "BINARY", "NOMINAL", "ORDINAL"], [continuous_count, discrete_count, binary_count, nominal_count, ordinal_count]), start=1):
                    #     st.write(f"{i}.&nbsp; <font color='blue'>{name}:</font> {count}", unsafe_allow_html=True)  
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
                                
        # st.sidebar.divider()
        # st.sidebar.title("🔍 RAW Data Overview:") 
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
                    st.subheader("[🔍] RAW Data Insights:")
                    st_profile_report(profile) 
                    file_name = f"raw_data_overview_{file.name.split('.')[0]}.html"
                    profile.to_file(file_name)

                    download_csv(file_name)        
                    remove_file(file_name)

if __name__ == '__main__':
    main()
