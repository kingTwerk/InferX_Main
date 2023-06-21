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
import two_way_anova_test

from scipy.stats import chi2_contingency
from scipy.stats import chi
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

import base64
import webbrowser
import time

from PIL import Image

st.set_page_config(page_title="INFER-X (Local-updated 042523)", layout='wide', initial_sidebar_state='expanded', page_icon="👁️‍🗨️")


lottie_hacking = load_lottiefile("lottiefiles/hacker.json")
lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_yalmtkoy.json")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("sidebar_rotate.jpg")
# top - bottom - left - right - top left - top right - bottom left - bottom right
page_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: top left; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.title("🕵🏽 Inferential Statistical Tests") 
# image = Image.open("test.png")
# st.image(image, use_column_width=True)
# st.markdown(""" <style> .font {                                          
# font-size:30px ; font-family: 'Cooper Black'; color: #FF9620;} 
# </style> """, unsafe_allow_html=True)
# st.markdown('<p class="font">Inferential Statistical Tests Recommender</p>', unsafe_allow_html=True)

def main():
    file = st.file_uploader("FYI: It is recommended to upload your data in csv format, it will help lessen the processing time.", type=["csv", "xlsx"])

    if file:
        show_sidebar = True
    else:
        show_sidebar = False
        st.markdown("🚧 Before uploading your data file to Infer-X, it's essential to ensure that your data is clean and well-prepared. Follow these simple steps to clean your data:")
        reminderA, reminderB, reminderC, reminderD, reminderE, reminderF = st.columns((2.5,2.5,2.5,2.5,2.5,2.5), gap="small")
        with reminderA:
            st.warning("1️⃣ Remove any unnecessary columns:")
            st.write("- Review your dataset and eliminate any columns that are irrelevant or contain no useful information for your analysis.")
        with reminderB:    
            st.warning("2️⃣ Check for missing values:")
            st.write("- Identify and handle missing values in your dataset. You can either delete rows with missing data or impute values based on specific rules or techniques.")
        with reminderC:
            st.warning("3️⃣ Address data inconsistencies:")
            st.write("- Look out for inconsistent formatting, spelling errors, or different representations of the same category. Standardize your data to ensure consistency and accuracy.")
        with reminderD:
            st.warning("4️⃣ Handle outliers:")
            st.write("- Identify any extreme or erroneous values that may significantly impact your analysis. Consider removing outliers or applying appropriate techniques to handle them.")
        with reminderE:
            st.warning("5️⃣ Validate data types:")
            st.write("-  Ensure that variables are assigned the correct data types (e.g., numeric, categorical, date) to avoid data interpretation issues during analysis. ")
        with reminderF:
            st.warning("6️⃣ Resolve duplicates:")
            st.write("- Identify and eliminate any duplicate records or entries in your dataset to prevent skewing your results.")
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

            st.sidebar.title("👟 Infer-X Steps:") 
            # st.sidebar.image(image, use_column_width=True)
            object_cols = df.select_dtypes(include=['object']).columns
            int_cols = df.select_dtypes(include=['int64']).columns
            float_cols = df.select_dtypes(include=['float64']).columns
            date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

            # Specify integer data type explicitly for integer columns
            # df[int_cols] = df[int_cols].astype('int64')            

            df[object_cols] = df[object_cols].fillna('Missing')
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
                "Continuous": ["ONE-WAY ANOVA", "TWO-WAY ANOVA", "SIMPLE LINEAR REGRESSION"],
                "Discrete": ["ONE-WAY ANOVA", "TWO-WAY ANOVA", "SIMPLE LINEAR REGRESSION"],
                "Binary": ["LOGISTIC REGRESSION", "CHI-SQUARE"],
                "Ordinal": ["CHI-SQUARE"],
                "Nominal": ["CHI-SQUARE"],
            }

            recommendations = {col: level_to_recommendation[level] for col, level in zip(df_final.columns, levels)}

            test_to_count = {
                "ONE-WAY ANOVA": 0,
                "TWO-WAY ANOVA": 0,
                "SIMPLE LINEAR REGRESSION": 0,
                "LOGISTIC REGRESSION": 0,
                "CHI-SQUARE": 0,
            }

            for tests in recommendations.values():
                for test in tests:
                    test_to_count[test] += 1

            anova_count = test_to_count["ONE-WAY ANOVA"]
            twowayanova_count = test_to_count["TWO-WAY ANOVA"]
            single_linear_regression_count = test_to_count["SIMPLE LINEAR REGRESSION"]
            logistic_regression_count = test_to_count["LOGISTIC REGRESSION"]
            chi_square_count = test_to_count["CHI-SQUARE"]
            
            column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(df_final.columns, isNumerical)]
            column_options = [f"{col}: {level}" for col, level in zip(df_final.columns, levels)]
            
            # colored_header(
            #     label="",
            #     description="",
            #     color_name="violet-70",
            # )   
            
            st.sidebar.write("1️⃣ FILTER BY COLUMN:")
            filter_checkbox = st.sidebar.checkbox("Filter (to exclude values)",key="checkbox3")
            
            if filter_checkbox: 
                if filter_checkbox:
                    df_final = filter_columns(df_final, filter_checkbox)
            else:
                df_final = df.copy()   
            st.sidebar.write("")    
            

            index2 = 0
            column = st.sidebar.selectbox("2️⃣ SELECT THE 'y' FIELD (dependent variable):", [''] + list(recommendations.keys()), index=index2)
    
            st.sidebar.write("")           
            if column:
                test = st.sidebar.selectbox("3️⃣ SELECT AN INFERENTIAL TEST (determined by the selected 'y' field above):", [''] + recommendations[column], index=0)
                st.sidebar.write("")
                transformation_check(df_final, isNumerical, column, test)
                
                if test == "SIMPLE LINEAR REGRESSION":
                    linear_regression_test.linear_regression(df_final,file,column)
                # elif test == "T-TEST (PAIRED)":
                #     anova_test.t_test_paired(df_final,file,column)
                elif test == "ONE-WAY ANOVA":
                    anova_test.anova(df_final,file,column)
                elif test == "CHI-SQUARE":
                    chi_square_test.chi_square(df_final,file,column,test)
                elif test == "LOGISTIC REGRESSION":
                    logistic_regression_test.logistic_regression(df_final,file,column)
                elif test == "TWO-WAY ANOVA":
                    two_way_anova_test.twoanova(df_final,file,column)
                else:
                    st.write("")
            else:
                test = None
            st.sidebar.write("\n")                        
            details_checkbox = st.sidebar.checkbox("Do you want to more details?", key="checkbox0")

            if details_checkbox:
                    colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )   
                    st.subheader("[🔍] Variable Insights:")
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["Ⅰ.┊NEEDS NORMALIZATION┊","Ⅱ.┊MEASUREMENT TYPES┊","Ⅲ.┊TEST SUGGESTIONS┊", "Ⅳ.┊LEVELS OF MEASUREMENTS┊", "Ⅴ.┊VARIABLE TYPES┊","Ⅵ.┊ALL TEST SUGGESTIONS┊","Ⅶ.┊UNIQUE VARIABLE┊"])
                    if not column:
                        st.write("")
                    #with tab8:
                    #    for i, (col_name, dtype) in enumerate(df_final.dtypes.iteritems(), start=1):
                    #        st.write(f"{i}.&nbsp; <font color='blue'>{col_name}</font>: {dtype}", unsafe_allow_html=True)
                        
                    with tab7:
                        for i, option in enumerate(column_options, start=1):
                            name, dtype = option.split(":")
                            st.write(f"{i}.&nbsp; <font color='blue'>{name.strip()} ({dtype.strip()}):</font> {df_final[name.strip()].nunique()} unique values", unsafe_allow_html=True)
                    with tab2:
                        for i, (name, dtype, count) in enumerate(zip(["CONTINUOUS", "DISCRETE", "BINARY", "NOMINAL", "ORDINAL"], ["float64", "int64", "int64", "object", "object"], [continuous_count, discrete_count, binary_count, nominal_count, ordinal_count]), start=1):
                            st.write(f"{i}.&nbsp; <font color='blue'>{name} ({dtype}):</font> {count}", unsafe_allow_html=True)                     
                    with tab3:
                        for i, (test_name, count) in enumerate(zip(['ONE-WAY ANOVA', 'TWO-WAY ANOVA', 'SIMPLE LINEAR REGRESSION', 'LOGISTIC REGRESSION', 'CHI-SQUARE'], [anova_count, twowayanova_count, single_linear_regression_count, logistic_regression_count, chi_square_count]), start=1):
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
            
                        needsNormalization = []
                        for column in df.columns:
                            if df[column].dtype == np.int64 or df[column].dtype == np.float64:
                                z_scores = (df[column] - df[column].mean()) / df[column].std()
                                if (z_scores.max() - z_scores.min()) > 3: 
                                    needsNormalization.append((column, z_scores))

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

            # image2 = Image.open("statistics.jpg")
            # st.sidebar.markdown(
            #     f"""
            #     <style>
            #     .sidebar .sidebar-content {{
            #         max-width: 100px;  /* Set the maximum width of the sidebar content */
            #     }}
            #     .sidebar .sidebar-content img {{
            #         margin-left: auto;
            #         margin-right: auto;
            #         display: block;
            #         height: 100px;
            #     }}
            #     </style>
            #     """,
            #     unsafe_allow_html=True,
            # )
            # st.sidebar.image(image2,  use_column_width="auto")

if __name__ == '__main__':
    main()
