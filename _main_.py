import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import chi2_contingency
from scipy.stats import chi
import linear_regression_test
import anova_test
import chi_square_test
import logistic_regression_test

st.set_page_config(page_title="INFER-X", layout='wide', initial_sidebar_state='expanded', page_icon='👁️‍🗨️')

st.title("👨🏽‍💻INFER-X") 
st.subheader("[Inferential Statistical Tests Recommender]")

file = st.file_uploader("➕ Choose a file", type=["csv", "xlsx"])

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
            data = pd.read_csv(file, na_values=["*","**","", "NA", "N/A","-"])
            st.subheader("👁️‍🗨️ Data Preview:")
            st.write("\n")
            if data.shape[0] == 0:
                st.warning("The selected file does not contain any data or all rows with blank / alike will be automatically removed.")
            else:
                csv_orig_row, csv_orig_col = st.columns((1,4), gap="small")
                with csv_orig_row:
                    st.write(f"Original number of rows :{data.shape[0]:.0f}")
                with csv_orig_col:
                    st.write(f"Original number of columns :{data.shape[1]:.0f}")
                    
                data = data.dropna() 
                st.dataframe(data)
                
                csv_drop_row, csv_drop_col = st.columns((1,4), gap="small")
                with csv_drop_row:
                    st.write(f"Currnet number of rows :{data.shape[0]:.0f}")
                with csv_drop_col:
                    st.write(f"Currnet number of columns :{data.shape[1]:.0f}")
                
            st.write("\n")
            
        except MemoryError:
            st.write("Mem Error")
            error_caught = True
            fixed_data = fix_dataframe(data)
            fixed_data = fixed_data.dropna() 
            st.dataframe(fixed_data)
            csv_fix_row, csv_fix_col = st.columns((1,4), gap="small")
            with csv_fix_row:
                st.write(f"Currnet number of rows :{data.shape[0]:.0f}")
            with csv_fix_col:
                st.write(f"Currnet number of columns :{data.shape[1]:.0f}")
        
    elif file_name.endswith(".xlsx"):
        xlsx_file = pd.read_excel(file, sheet_name=None, na_values=["*","**","", "NA", "N/A","-"])
        sheet_names = list(xlsx_file.keys())
        
        selected_sheet = st.selectbox("➕ Select a sheet:", sheet_names)
        
        try:
            data = pd.read_excel(file, sheet_name=selected_sheet, na_values=["*","**","", "NA", "N/A","-"]) 
            st.write("👁️‍🗨️ Data Preview:")
            st.write("\n")
            if data.shape[0] == 0:
                st.warning("The selected file does not contain any data or all rows with blank / alike will be automatically removed.")
            else:
                xl_orig_row, xl_orig_col = st.columns((1,4), gap="small")
                with xl_orig_row:
                    st.write(f"Original number of rows :{data.shape[0]:.0f}")
                with xl_orig_col:
                    st.write(f"Original number of columns :{data.shape[1]:.0f}")
                    
                data = data.dropna() 
                st.dataframe(data)
                
                xl_drop_row, xl_drop_col = st.columns((1,4), gap="small")
                with xl_drop_row:
                    st.write(f"Current number of rows :{data.shape[0]:.0f}")
                with xl_drop_col:
                    st.write(f"Current number of columns :{data.shape[1]:.0f}")
                    
            st.write("\n")
        
        except MemoryError:
            st.write("Mem Error")
            error_caught = True
            fixed_data = fix_dataframe(data)
            fixed_data = fixed_data.dropna() 
            st.dataframe(fixed_data)
            xl_fix_row, xl_fix_col = st.columns((1,4), gap="small")
            with xl_fix_row:
                st.write(f"Current number of rows :{data.shape[0]:.0f}")
            with xl_fix_col:
                st.write(f"Current number of columns :{data.shape[1]:.0f}")

    if error_caught:
        #st.warning("There was not enough memory to process the data. Please try again with a smaller dataset.")
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
        isNumerical = []
        for columndata in data.columns:
            if data[columndata].dtype == np.int64 or data[columndata].dtype == np.float64:
                isNumerical.append("Quantitative")
            else:
                isNumerical.append("Categorical")
                
        column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(data.columns, isNumerical)]
        column_options = [f"{col}: {level}" for col, level in zip(data.columns, levels)]

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

        st.sidebar.markdown("<i>RECOMMENDATIONS PER COLUMN:</i>", unsafe_allow_html=True)
        for column, tests in recommendations.items():
            tbl_column = f"<font style='font-size:15pt;font-family:Courier;color:purple' unsafe_allow_html=True><b>- {column}:</b></font>"
            tbl_test = f"<font style='font-size:10pt;font-family:Courier;color:#00a67d'><b>{', '.join(tests)}</b></font>" 
            st.sidebar.markdown(tbl_column + tbl_test, unsafe_allow_html=True)

        if data.empty:
            st.write("")
        else:
            column = st.selectbox("➕ Choose a column:", list(recommendations.keys()))
            col1, col2, col3 = st.columns(3)
            if not column:
                st.write("No column selected")
            with col1:
                st.write(f"🔽 Levels of Measurements per column :", column_options, key=f"test-{column}")
            with col2:    
                st.write(f"🔽 Variable column type:", column_numerical, key=f"test-{column}")
            with col3:
                st.write(f"🔽 Test suggestion for column {column}:", recommendations[column], key=f"test-{column}")
            
            test = st.selectbox("➕ Choose a test:", recommendations[column])
            
            if test == "SINGLE LINEAR REGRESSION":
                linear_regression_test.linear_regression(data,file)
            elif test == "T-TEST (PAIRED)":
                anova_test.t_test_paired(data,file)
            elif test == "ANOVA":
                anova_test.anova(data,file)
            elif test == "CHI-SQUARE TEST":
                chi_square_test.chi_square_fixed(fixed_data,col1,col2)
            elif test == "LOGISTIC REGRESSION":
                logistic_regression_test.logistic_regression(data,file)
        
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
        isNumerical = []
        for columndata in data.columns:
            if data[columndata].dtype == np.int64 or data[columndata].dtype == np.float64:
                isNumerical.append("Quantitative")
            else:
                isNumerical.append("Categorical")
                
        column_numerical = [f"{columndata}: {isNumerical}" for columndata, isNumerical in zip(data.columns, isNumerical)]
        column_options = [f"{col}: {level}" for col, level in zip(data.columns, levels)]

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

        st.sidebar.markdown("<i>RECOMMENDATIONS PER COLUMN:</i>", unsafe_allow_html=True)
        for column, tests in recommendations.items():
            tbl_column = f"<font style='font-size:15pt;font-family:Courier;color:purple' unsafe_allow_html=True><b>- {column}:</b></font>"
            tbl_test = f"<font style='font-size:10pt;font-family:Courier;color:#00a67d'><b>{', '.join(tests)}</b></font>" 
            st.sidebar.markdown(tbl_column + tbl_test, unsafe_allow_html=True)

        if data.empty:
            st.write("")
        else:
            column = st.selectbox("➕ Choose a column:", list(recommendations.keys()))
            col1, col2, col3 = st.columns(3)
            if not column:
                st.write("No column selected")
            with col1:
                st.write(f"🔽 Levels of Measurements per column :", column_options, key=f"test-{column}")
            with col2:    
                st.write(f"🔽 Variable column type:", column_numerical, key=f"test-{column}")
            with col3:
                st.write(f"🔽 Test suggestion for column {column}:", recommendations[column], key=f"test-{column}")
            
            test = st.selectbox("➕ Choose a test:", recommendations[column])
            
            if test == "SINGLE LINEAR REGRESSION":
                linear_regression_test.linear_regression(data,file)
            elif test == "T-TEST (PAIRED)":
                anova_test.t_test_paired(data,file)
            elif test == "ANOVA":
                anova_test.anova(data,file)
            elif test == "CHI-SQUARE TEST":
                chi_square_test.chi_square(data,col1,col2)
            elif test == "LOGISTIC REGRESSION":
                logistic_regression_test.logistic_regression(data,file)
            