import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.colored_header import colored_header
from scipy.stats import f
from anova_results import anova_result
from functions import is_ordinal, normalize_numpy, transform_column

def anova(df_final, file, column):

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header('🧮 Analysis of variance (ANOVA)')
        st.write("\n")
        
        with st.expander("What is One-way ANOVA?",expanded=True):   
            st.write("The test is used to determine if there is a significant difference in the means of the dependent variable across the different levels of the independent variable.")
            st.markdown("- y: a quantitative variable (e.g., height, age)")
            st.markdown("- x: a categorical variable with at least three levels (e.g., class A, class B, class C)")

            st.write("")        
            st.markdown('''
                <style>
                [data-testid="stMarkdownContainer"] ul{
                    padding-left:40px;
                }
                </style>
                ''', unsafe_allow_html=True)

        column_names = df_final.columns.tolist()

        column_names = [col for col in column_names if df_final[col].dtype == 'object']
        independent_column_name = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", column_names, key='anova_column')
        
        dependent_column_name = column
        
        if independent_column_name == dependent_column_name:
            st.error("❌ Both columns are the same. Please select different columns.")
        else:           
            
                if (not np.issubdtype(df_final[dependent_column_name], np.number) and df_final[dependent_column_name].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #1: {dependent_column_name} column might contain categorical/string variables, column can be either discrete or continuous data with atleast 2 unique values.')
                
                elif (df_final[independent_column_name].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #2: {independent_column_name} column must have atleast 2 unique values.') 
                
                elif (df_final[dependent_column_name].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #3: {dependent_column_name} column must have atleast 2 unique values.') 
                
                elif (pd.api.types.is_string_dtype(df_final[independent_column_name])):

                    numeric_cols = df_final._get_numeric_data().columns
                    categorical_cols = df_final.columns.difference(numeric_cols)
                    num_cols = df_final.select_dtypes(include=['int64', 'float64']).columns.tolist()

                    needs_normalization = []
                    for col in numeric_cols:
                        z_scores = (df_final[col] - df_final[col].mean()) / df_final[col].std()
                        if (z_scores.max() - z_scores.min()) > 3: 
                            needs_normalization.append(col)

                    common_cols = set([independent_column_name, dependent_column_name]).intersection(set(needs_normalization))

                    if common_cols:
                        default_values = list(common_cols)
                    else:
                        default_values = []

                    selected_cols = st.sidebar.multiselect("👉 COLUMN TO BE NORMALIZED (for selected 'y' field above):", needs_normalization, default=default_values)

                    df_final = df_final.copy()
      
                    if len(selected_cols) > 0:
                        method = "Z-Score"
                        numeric_selected_cols = [col for col in selected_cols if col in numeric_cols]
                        categorical_selected_cols = [col for col in selected_cols if col not in numeric_cols]
                        df_norm = normalize_numpy(df_final, numeric_selected_cols, categorical_selected_cols, method)
                        not_selected_cols = df_final.columns.difference(selected_cols)
                        df_final = pd.concat([df_norm, df_final[not_selected_cols]], axis=1)

                    levels = np.empty(df_final.shape[1], dtype=object) 
                    for i, col in enumerate(df_final.columns):  
                        if df_final[col].dtype == 'bool' or df_final[col].dtype == 'category':
                            levels[i] = "Binary"
                        elif df_final[col].dtype == np.int64:
                            levels[i] = "Discrete"
                        elif df_final[col].dtype == np.float64:
                            levels[i] = "Continuous"
                        elif df_final[col].dtype == object:
                            if is_ordinal(df_final[col]):
                                levels[i] = "Ordinal"
                            else:
                                levels[i] = "Nominal"

                    independent_column_index = df_final.columns.get_loc(independent_column_name)
                    if levels[independent_column_index] == "Nominal" and len(df_final[independent_column_name].unique()) > 2:
                        recommended_method = "One-Hot"
                    elif levels[independent_column_index] == "Binary" and len(df_final[independent_column_name].unique()) == 2:
                        recommended_method = "Label"
                    elif levels[independent_column_index] == "Ordinal":
                        recommended_method = "Ordinal"
                    else:
                        recommended_method = None

                    if recommended_method:
                        method = recommended_method
                    else:
                        method = "Label"
                    transformed_col = transform_column(df_final, independent_column_name, method)

                    df_final[independent_column_name] = transformed_col

                    st.subheader("[👁️‍🗨️] Data Preview:")
                    st.dataframe(df_final, height = 400)

                    button,anova_row, anova_col = st.columns((0.0001,1.5,4.5), gap="small")
                    rows = df_final.shape[0]
                    cols = df_final.shape[1]
                    with anova_row:
                        st.markdown(f"<span style='color: violet;'>➕ # of rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
                    with anova_col:
                        st.markdown(f"<span style='color: violet;'>➕ # of columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)

                    colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )     
                    
                    anova_result(df_final, independent_column_name, dependent_column_name, file, column)
                    
                elif (pd.api.types.is_string_dtype(df_final[dependent_column_name])):
                    st.error(f'❌ SELECTION ERROR #5: {dependent_column_name} column might contain categorical/string variables, please select a quantitative column.')
