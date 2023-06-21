import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.colored_header import colored_header
from scipy.stats import f
from two_way_anova_test_results import two_anova_result
from functions import is_ordinal, normalize_numpy, transform_column

def twoanova(df_final, file, column):

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header('🧮 Two-Way ANOVA')
        st.write("\n")
        
        with st.expander("What is Two-way ANOVA?", expanded=True):   
            st.write("Two-way ANOVA is a statistical test used to analyze the effects of two categorical independent variables (factors) on a numerical dependent variable. It examines whether there are significant differences in the means of the dependent variable across different levels of each independent variable, as well as the interaction between the two independent variables.")

            st.markdown("In Two-way ANOVA:")
            st.markdown("- The dependent variable (y) is a numerical variable (e.g., height, age).")
            st.markdown("- The first independent variable (x1) is a categorical variable with at least two levels (e.g., treatment groups A and B).")
            st.markdown("- The second independent variable (x2) is another categorical variable with at least two levels (e.g., gender categories Male and Female).")

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
        independent_column_name = st.sidebar.selectbox("4️⃣ SELECT THE 'x1' FIELD (first independent variable):", [""] + column_names, key='anova_column')
        column_names_x2 = [""] + [col for col in column_names if col != independent_column_name]
        independent_column_name2 = st.sidebar.selectbox("5️⃣ SELECT THE 'x2' FIELD (second independent variable):", column_names_x2, key='anova_column2')
        dependent_column_name = column
        
        if independent_column_name == dependent_column_name:
            st.error("❌ Both columns are the same. Please select different columns.")
        else:           
                if independent_column_name2 == "" or independent_column_name == "":
                    st.write("")
                elif (not np.issubdtype(df_final[dependent_column_name], np.number) and df_final[dependent_column_name].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #1: {dependent_column_name} column might contain categorical/string variables, column can be either discrete or continuous data with atleast 2 unique values.')
                
                elif (df_final[independent_column_name].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #2: {independent_column_name} column must have atleast 2 unique values.') 
                
                elif (df_final[dependent_column_name].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #3: {dependent_column_name} column must have atleast 2 unique values.') 
                
                elif (pd.api.types.is_string_dtype(df_final[independent_column_name])):

                    numeric_cols = df_final._get_numeric_data().columns

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
                        #st.write("👉 One-Hot Encoding is recommended for the selected 'x1' field.")
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

                    independent_column_index2 = df_final.columns.get_loc(independent_column_name2)
                    if levels[independent_column_index2] == "Nominal" and len(df_final[independent_column_name2].unique()) > 2:
                        recommended_method2 = "One-Hot"
                    elif levels[independent_column_index2] == "Binary" and len(df_final[independent_column_name2].unique()) == 2:
                        recommended_method2 = "Label"
                    elif levels[independent_column_index2] == "Ordinal":
                        recommended_method2 = "Ordinal"
                    else:
                        recommended_method2 = None

                    if recommended_method2:
                        method2 = recommended_method2
                    else:
                        method2 = "Label"

                    transformed_col2 = transform_column(df_final, independent_column_name2, method2)

                    df_final[independent_column_name2] = transformed_col2

                    st.subheader("[👁️‍🗨️] Data Preview:")
                    st.dataframe(df_final, height = 400)

                    button,anova_row, anova_col = st.columns((0.0001,1.5,4.5), gap="small")
                    rows = df_final.shape[0]
                    cols = df_final.shape[1]
                    with anova_row:
                        st.markdown(f"<span style='color: violet;'>➕ Number of rows : </span> <span style='color: #D3D3D3;'>{rows}</span>", unsafe_allow_html=True)
                    with anova_col:
                        st.markdown(f"<span style='color: violet;'>➕ Number of columns : </span> <span style='color: #D3D3D3;'>{cols}</span>", unsafe_allow_html=True)

                    colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )     
                    
                    two_anova_result(df_final, independent_column_name, independent_column_name2, dependent_column_name, file, column)
                    
                elif (pd.api.types.is_string_dtype(df_final[dependent_column_name])):
                    st.error(f'❌ SELECTION ERROR #5: {dependent_column_name} column might contain categorical/string variables, please select a quantitative column.')
