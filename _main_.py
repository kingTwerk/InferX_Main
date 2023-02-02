import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import chi2_contingency
from scipy.stats import chi
from streamlit_extras.colored_header import colored_header

import os
import datetime

import linear_regression_test
import anova_test
import chi_square_test
import logistic_regression_test

st.set_page_config(page_title="INFER-X-sidebar", layout='wide', initial_sidebar_state='expanded', page_icon='👁️‍🗨️')

st.sidebar.title("⚙ INFER-X") 
st.title("🖊️ Inferential Statistical Tests Recommender")

file = st.sidebar.file_uploader("➕ UPLOAD A FILE", type=["csv", "xlsx"])

@st.experimental_memo
def fix_dataframe(df):
    fixed_df = df.copy()
    for col in fixed_df.columns:
        try:
            _ = fixed_df[col].apply(bytes)
        except (TypeError, ValueError):
            fixed_df[col] = fixed_df[col].apply(str)
    return fixed_df

error_caught = False

if file is not None:
    file_name = file.name
    file.seek(0)

    if file_name.endswith(".csv"):
        try:
            data = pd.read_csv(file) 
            #csv_orig_row, csv_orig_col = st.columns((5,1), gap="small")
            original_rows = data.shape[0]
            original_cols = data.shape[1]
            
            #with csv_orig_row:
            st.sidebar.markdown(f"<span style='color: blue ;'>Original number of rows : </span> <span style='color: black;'>{original_rows}</span>", unsafe_allow_html=True)
            #with csv_orig_col:
            st.sidebar.markdown(f"<span style='color: blue;'>Original number of columns : </span> <span style='color: black;'>{original_cols}</span>", unsafe_allow_html=True)
            
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )  

            st.subheader("[👁️‍🗨️] Table Preview:")
            st.write("\n")
            if data.shape[0] == 0:
                st.warning("The selected file does not contain any data or all rows with blank / alike will be automatically removed.")
            else:
                st.dataframe(data)
                space, csv_drop_row, csv_drop_col = st.columns((5,1,1), gap="small")
                drop_rows = data.shape[0]
                drop_cols = data.shape[1]

                with csv_drop_row:
                    st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{drop_rows}</span>", unsafe_allow_html=True)
                with csv_drop_col:
                    st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{drop_cols}</span>", unsafe_allow_html=True)
            
        except MemoryError:
            st.write("Mem Error")
            error_caught = True
            fixed_data = fix_dataframe(data)
            fixed_data = fixed_data.dropna() 
            st.dataframe(fixed_data)           

            space,csv_fix_row, csv_fix_col = st.columns((5,1,1), gap="small")
            fxd_rows = data.shape[0]
            fxd_cols = data.shape[1]
            #st.write("Count after dropping records with N/A or blank records:")
            with csv_fix_row:
                st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{fxd_rows}</span>", unsafe_allow_html=True)
            with csv_fix_col:
                st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{fxd_cols}</span>", unsafe_allow_html=True)
        
    elif file_name.endswith(".xlsx"):
        #xlsx_file = pd.read_excel(file, sheet_name=None, na_values=["*","**","", "NA", "N/A","-"])
        xlsx_file = pd.read_excel(file, sheet_name=None)
        sheet_names = list(xlsx_file.keys())
        st.sidebar.markdown("""---""")
        selected_sheet = st.sidebar.selectbox("➕ SELECT A SHEET", sheet_names)
        
        try:
            data = pd.read_excel(file, sheet_name=selected_sheet)
       
            original_rows = data.shape[0]
            original_cols = data.shape[1]
          
            st.sidebar.markdown(f"<span style='color: blue;'>Original number of rows : </span> <span style='color: black;'>{original_rows}</span>", unsafe_allow_html=True)
         
            st.sidebar.markdown(f"<span style='color: blue;'>Original number of columns : </span> <span style='color: black;'>{original_cols}</span>", unsafe_allow_html=True)
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )              
            st.subheader("[👁️‍🗨️] Table Preview:")
            st.write("\n")
            if data.shape[0] == 0:
                st.warning("The selected file does not contain any data or all rows with blank / alike will be automatically removed.")
            else:   
                
                st.dataframe(data)

                xl_drop_row, xl_drop_col = st.columns((1,5), gap="small")
                xldrop_rows = data.shape[0]
                xldrop_cols = data.shape[1]
     
                with xl_drop_row:
                    st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{xldrop_rows}</span>", unsafe_allow_html=True)
                with xl_drop_col:
                    st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{xldrop_cols}</span>", unsafe_allow_html=True)
                    
            st.write("\n")
        
        except MemoryError:
            st.write("Mem Error")
            error_caught = True
            fixed_data = fix_dataframe(data)
            fixed_data = fixed_data.dropna() 
            st.dataframe(fixed_data)
 
            button, xl_fix_row, xl_fix_col = st.columns((5,1,1), gap="small")
            xlfix_rows = data.shape[0]
            xlfix_cols = data.shape[1]

            with xl_fix_row:
                st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{xlfix_rows}</span>", unsafe_allow_html=True)
            with xl_fix_col:
                st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{xlfix_cols}</span>", unsafe_allow_html=True)
                
    if error_caught:
 
        st.dataframe(data)

        @st.experimental_memo
        def is_ordinal(data):
            try:
                cat_data = pd.Categorical(data)
                return cat_data.ordered
            except ValueError:
                return False
            
        levels = np.empty(data.shape[1], dtype=object) 
        for i, col in enumerate(data.columns):  
            if data[col].dtype == np.int64 or data[col].dtype == np.float64:
                if data[col].dtype == np.int64:
                    levels[i] = "Discrete"
                else:
                    levels[i] = "Continuous"
            elif data[col].dtype == object:
                if data[col].nunique() <= 2:
                    levels[i] = "Binary"
                else:
                    if is_ordinal(data[col]):
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
        for columndata in data.columns:
            if data[columndata].dtype == np.int64 or data[columndata].dtype == np.float64:
                isNumerical.append("Quantitative")
            else:
                isNumerical.append("Categorical")

        recommendations = {}
        for i, level in enumerate(levels):
            if level == "Continuous":
                recommendations[data.columns[i]] = ["ANOVA", "SINGLE LINEAR REGRESSION"]
            elif level == "Binary":
                recommendations[data.columns[i]] = ["ANOVA","CHI-SQUARE TEST", "LOGISTIC REGRESSION"]
            elif level == "Ordinal":
                recommendations[data.columns[i]] = ["ANOVA", "CHI-SQUARE TEST"]
            elif level == "Discrete":
                recommendations[data.columns[i]] = ["ANOVA","CHI-SQUARE TEST", "LOGISTIC REGRESSION"]
            elif level == "Nominal":
                recommendations[data.columns[i]] = ["ANOVA","CHI-SQUARE TEST"]
            else:
                recommendations[data.columns[i]] = ["NO RECOMMENDATION"]

        anova_count = 0
        single_linear_regression_count = 0
        logistic_regression_count = 0
        chi_square_count = 0

        for tests in recommendations.values():
            for test in tests:
                if test == "ANOVA":
                    anova_count += 1
                elif test == "SINGLE LINEAR REGRESSION":
                    single_linear_regression_count += 1
                elif test == "LOGISTIC REGRESSION":
                    logistic_regression_count += 1
                elif test == "CHI-SQUARE TEST":
                    chi_square_count += 1

        if data.empty:
            st.write("")
        else:
            column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(data.columns, isNumerical)]
            column_options = [f"{col}: {level}" for col, level in zip(data.columns, levels)]
            
            tab1, tab2, tab3, tab4, tab5, tab6  = st.tabs(["🔽  CURRENT COLUMN TEST SUGGESTIONS  |","🔽  LEVELS OF MEASUREMENTS COUNT  |","🔽  TEST SUGGESTIONS COUNT  |", "🔽  LEVELS OF MEASUREMENTS  |", "🔽  VARIABLE COLUMN TYPE  |","🔽  ALL COLUMNSS TEST SUGGESTIONS  |"])
            
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )   
            st.sidebar.markdown("""---""")
            column = st.sidebar.selectbox("➕ SELECT A COLUMN:", list(recommendations.keys()))
            st.write("\n")
            test = st.sidebar.selectbox("➕ SELECT A TEST:", recommendations[column])

            if test == "SINGLE LINEAR REGRESSION":
                linear_regression_test.linear_regression(data,file)
            elif test == "T-TEST (PAIRED)":
                anova_test.t_test_paired(data,file)
            elif test == "ANOVA":
                anova_test.anova(data,file)
            elif test == "CHI-SQUARE TEST":
                chi_square_test.chi_square(data,file)
            elif test == "LOGISTIC REGRESSION":
                logistic_regression_test.logistic_regression(data,file)
            else:
                st.write("Invalid test selected")

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
                for i, (test_name, count) in enumerate(zip(['ANOVA', 'SINGLE LINEAR REGRESSION', 'LOGISTIC REGRESSION', 'CHI-SQUARE TEST'], [anova_count, single_linear_regression_count, logistic_regression_count, chi_square_count]), start=1):
                    st.write(f"{i}. <font color='blue'>{test_name}:</font> {count}", unsafe_allow_html=True)
    else:
        @st.experimental_memo
        def is_ordinal(fixed_data):
            try:
                cat_data = pd.Categorical(fixed_data)
                return cat_data.ordered
            except ValueError:
                return False
            
        levels = np.empty(data.shape[1], dtype=object) 
        for i, col in enumerate(data.columns):  
            if data[col].dtype == np.int64 or data[col].dtype == np.float64:
                if data[col].dtype == np.int64:
                    levels[i] = "Discrete"
                else:
                    levels[i] = "Continuous"
            elif data[col].dtype == object:
                if data[col].nunique() <= 2:
                    levels[i] = "Binary"
                else:
                    if is_ordinal(data[col]):
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
        for columndata in data.columns:
            if data[columndata].dtype == np.int64 or data[columndata].dtype == np.float64:
                isNumerical.append("Quantitative")
            else:
                isNumerical.append("Categorical")

        recommendations = {}
        for i, level in enumerate(levels):
            if level == "Continuous":
                recommendations[data.columns[i]] = ["ANOVA", "SINGLE LINEAR REGRESSION"]
            elif level == "Binary":
                recommendations[data.columns[i]] = ["ANOVA","CHI-SQUARE TEST", "LOGISTIC REGRESSION"]
            elif level == "Ordinal":
                recommendations[data.columns[i]] = ["ANOVA", "CHI-SQUARE TEST"]
            elif level == "Discrete":
                recommendations[data.columns[i]] = ["ANOVA","CHI-SQUARE TEST", "LOGISTIC REGRESSION"]
            elif level == "Nominal":
                recommendations[data.columns[i]] = ["ANOVA","CHI-SQUARE TEST"]
            else:
                recommendations[data.columns[i]] = ["NO RECOMMENDATION"]

        anova_count = 0
        single_linear_regression_count = 0
        logistic_regression_count = 0
        chi_square_count = 0

        for tests in recommendations.values():
            for test in tests:
                if test == "ANOVA":
                    anova_count += 1
                elif test == "SINGLE LINEAR REGRESSION":
                    single_linear_regression_count += 1
                elif test == "LOGISTIC REGRESSION":
                    logistic_regression_count += 1
                elif test == "CHI-SQUARE TEST":
                    chi_square_count += 1

        if data.empty:
            st.write("")
        else:
            column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(data.columns, isNumerical)]
            column_options = [f"{col}: {level}" for col, level in zip(data.columns, levels)]
            
            tab1, tab2, tab3, tab4, tab5, tab6  = st.tabs(["🔽  CURRENT COLUMN TEST SUGGESTIONS  |","🔽  LEVELS OF MEASUREMENTS COUNT  |","🔽  TEST SUGGESTIONS COUNT  |", "🔽  LEVELS OF MEASUREMENTS  |", "🔽  VARIABLE COLUMN TYPE  |","🔽  ALL COLUMNSS TEST SUGGESTIONS  |"])

            
            colored_header(
                label="",
                description="",
                color_name="violet-70",
            )   
            st.sidebar.markdown("""---""")
            column = st.sidebar.selectbox("➕ SELECT A COLUMN", list(recommendations.keys()))
            st.sidebar.write("\n")
            test = st.sidebar.selectbox("➕ SELECT A TEST", recommendations[column])

            if test == "SINGLE LINEAR REGRESSION":
                linear_regression_test.linear_regression(data,file)
            elif test == "T-TEST (PAIRED)":
                anova_test.t_test_paired(data,file)
            elif test == "ANOVA":
                anova_test.anova(data,file)
            elif test == "CHI-SQUARE TEST":
                chi_square_test.chi_square(data,file)
            elif test == "LOGISTIC REGRESSION":
                logistic_regression_test.logistic_regression(data,file)
            else:
                st.write("Invalid test selected")

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
                for i, (test_name, count) in enumerate(zip(['ANOVA', 'SINGLE LINEAR REGRESSION', 'LOGISTIC REGRESSION', 'CHI-SQUARE TEST'], [anova_count, single_linear_regression_count, logistic_regression_count, chi_square_count]), start=1):
                    st.write(f"{i}. <font color='blue'>{test_name}:</font> {count}", unsafe_allow_html=True)
