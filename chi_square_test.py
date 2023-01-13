import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency, chi2

def chi_square_fixed(fixed_data, col1, col2):
    #py.init_notebook_mode()

    # Select only categorical columns
    fixed_data = fixed_data.select_dtypes(include=['object'])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('📝 Chi-Square Test')
    # Read in the file
    
    st.dataframe(fixed_data)
    st.write(f"Number of rows :{fixed_data.shape[0]:.0f}")
    st.write(f"Number of columns :{fixed_data.shape[1]:.0f}")    
    # Get the list of column names
    column_names = fixed_data.columns.tolist()

    # Select the columns to encode and test
    st.write('Dataframe fixed.')
    independent_column_name  = st.selectbox("➕ Select the column name for the X (independent/CATEGORICAL/DISCRETE) variable", column_names)
    #column_2 = st.selectbox("➕ Select the second column (categorical):", column_names)
    dependent_column_name = st.selectbox("➕ Select the column name for the y (independent/CATEGORICAL/DISCRETE) variable", column_names)

    # Check that no two or three columns are the same
    # if independent_column_name  == column_2 or independent_column_name  == dependent_column_name or column_2 == dependent_column_name:
    if independent_column_name  == dependent_column_name:
        st.error("❌ Both columns are the same. Please select different columns.")
    else:           
        try:
            if ((not pd.api.types.is_string_dtype(fixed_data[independent_column_name]) and not pd.api.types.is_integer_dtype(fixed_data[independent_column_name])) and fixed_data[independent_column_name].nunique() < 2):
                st.error(f'❌ {independent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
            
            elif ((not pd.api.types.is_string_dtype(fixed_data[dependent_column_name]) and not pd.api.types.is_integer_dtype(fixed_data[dependent_column_name])) and not fixed_data[dependent_column_name].nunique() < 2):
                st.error(f'❌ {dependent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
            
            else:
                # Select the significance level
                #alpha = st.slider("Select the significance level (alpha):", 0.0, 1.0, 0.05)
                alpha = 0.05
    
                # Initialize the encoder
                encoder = OrdinalEncoder()

                # Encode the selected columns
                # fixed_data[[independent_column_name , column_2]] = encoder.fit_transform(fixed_data[[independent_column_name , column_2]])
                fixed_data[[independent_column_name ]] = encoder.fit_transform(fixed_data[[independent_column_name ]])
                
                # Compute the chi-square test for each column
                chi2_score_1, p_value_1, _, _ = chi2_contingency(pd.crosstab(fixed_data[independent_column_name ], fixed_data[dependent_column_name]))
                #chi2_score_2, p_value_2, _, _ = chi2_contingency(pd.crosstab(fixed_data[column_2], fixed_data[dependent_column_name]))

                # Compute the degrees of freedom and the critical value
                degrees_of_freedom_1 = fixed_data[independent_column_name ].nunique() - 1
                #degrees_of_freedom_2 = fixed_data[column_2].nunique() - 1
                critical_value_1 = chi2.ppf(q=1-alpha, df=degrees_of_freedom_1)
                #critical_value_2 = chi2.ppf(q=1-alpha, df=degrees_of_freedom_2)
                #st.write(f'```\nChi-Square Score: {chi2_score_1}\nP-Value: {p_value_1}\nDegrees of Freedom: {degrees_of_freedom_1}\nCritical Value: {critical_value_1}\n```')

                # Display the results
                st.subheader("✎ [Chi-Square Test]")
                chi_col1, chi_col2, chi_col3 = st.columns((.5,.5,3), gap="small")
                with chi_col1:
                    #st.write(f'Chi-Square Score: {chi2_score_1:.2f}',  prefix="\t")
                    st.metric("Chi-Square Score:",f"{chi2_score_1:.2f}")
                with chi_col2:
                    #st.write(f'Critical Value: {critical_value_1:.2f}', prefix="\t")
                    st.metric("Critical Value:",f"{critical_value_1:.2f}")
                with chi_col3:
                    #st.write(f'P-Value: {p_value_1:.2f}', prefix="\t")
                    st.metric("P-Value:",f"{p_value_1:.2f}")
                #st.write(f'Degrees of Freedom: {degrees_of_freedom_1:.2f}', prefix="\t")
                if chi2_score_1 > critical_value_1:
                    st.success(f"* Reject the null hypothesis - the observed frequencies are significantly different from the expected frequencies. This suggests that there is a relationship between the two categorical variables.")
                    if p_value_1 < 0.05:
                        st.success(f"* The p-value is less than 5%, which suggests that the relationship is statistically significant.")
                    else:
                        st.error(f"* The p-value is greater than or equal to 5%, which suggests that the relationship is not statistically significant.")
                else:
                    st.error(f"* Fail to reject the null hypothesis - the observed frequencies are not significantly different from the expected frequencies. This suggests that there is no relationship between the two categorical variables. The observed frequencies are not significantly different from the expected frequencies.")
                st.write("\n")
        
        except TypeError:
                st.error(f'{dependent_column_name} is categorical/discrete with at least 2 unique values and {independent_column_name} is continuous.')  


def chi_square(data, col1, col2):

    # Select only categorical columns
    data = data.select_dtypes(include=['object','int'])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('📝 Chi-Square Test')
    # Read in the file
    st.dataframe(data)
    st.write(f"Number of rows :{data.shape[0]:.0f}")
    st.write(f"Number of columns :{data.shape[1]:.0f}")       
    # Get the list of column names
    column_names = data.columns.tolist()

    # Select the columns to encode and test
    independent_column_name  = st.selectbox("➕ Select the column name for the X (independent/CATEGORICAL/DISCRETE) variable", column_names)
    #column_2 = st.selectbox("➕ Select the second column (categorical):", column_names)
    dependent_column_name = st.selectbox("➕ Select the column name for the y (independent/CATEGORICAL/DISCRETE) variable", column_names)

    # Check that no two or three columns are the same
    # if independent_column_name  == column_2 or independent_column_name  == dependent_column_name or column_2 == dependent_column_name:

    if independent_column_name  == dependent_column_name:
        st.error("⚠️ Both columns are the same. Please select different columns.")
    else:           
        try:
            if ((not pd.api.types.is_string_dtype(data[independent_column_name]) and not pd.api.types.is_integer_dtype(data[independent_column_name])) and data[independent_column_name].nunique() < 2):
                st.error(f'1. {independent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
            
            elif ((not pd.api.types.is_string_dtype(data[dependent_column_name]) and not pd.api.types.is_integer_dtype(data[dependent_column_name])) and not data[dependent_column_name].nunique() < 2):
                st.error(f'2. {dependent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
            
            elif (pd.api.types.is_float_dtype(data[independent_column_name]) and np.issubdtype(data[independent_column_name], np.number)):
                st.error(f'3. {independent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
            
            elif (pd.api.types.is_float_dtype(data[dependent_column_name]) and np.issubdtype(data[dependent_column_name], np.number)):
                st.error(f'4. {dependent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')

            else:
            
                # Select the significance level
                #alpha = st.slider("Select the significance level (alpha):", 0.0, 1.0, 0.05)
                alpha = 0.05

                # Initialize the encoder
                encoder = OrdinalEncoder()

                # Encode the selected columns
                # fixed_data[[independent_column_name , column_2]] = encoder.fit_transform(fixed_data[[independent_column_name , column_2]])
                data[[independent_column_name ]] = encoder.fit_transform(data[[independent_column_name ]])
                
                # Compute the chi-square test for each column
                chi2_score_1, p_value_1, _, _ = chi2_contingency(pd.crosstab(data[independent_column_name ], data[dependent_column_name]))
                #chi2_score_2, p_value_2, _, _ = chi2_contingency(pd.crosstab(fixed_data[column_2], fixed_data[dependent_column_name]))

                # Compute the degrees of freedom and the critical value
                degrees_of_freedom_1 = data[independent_column_name ].nunique() - 1
                #degrees_of_freedom_2 = fixed_data[column_2].nunique() - 1
                critical_value_1 = chi2.ppf(q=1-alpha, df=degrees_of_freedom_1)
                #critical_value_2 = chi2.ppf(q=1-alpha, df=degrees_of_freedom_2)

                # Display the results
                st.subheader("✎ [Chi-Square Test]")
                chi_col1, chi_col2, chi_col3 = st.columns((.5,.5,3), gap="small")
                with chi_col1:
                    #st.write(f'Chi-Square Score: {chi2_score_1:.2f}',  prefix="\t")
                    st.metric("Chi-Square Score:",f"{chi2_score_1:.2f}")
                with chi_col2:
                    #st.write(f'Critical Value: {critical_value_1:.2f}', prefix="\t")
                    st.metric("Critical Value:",f"{critical_value_1:.2f}")
                with chi_col3:
                    #st.write(f'P-Value: {p_value_1:.2f}', prefix="\t")
                    st.metric("P-Value:",f"{p_value_1:.2f}")
                #st.write(f'Degrees of Freedom: {degrees_of_freedom_1:.2f}', prefix="\t")
                
                if chi2_score_1 > critical_value_1:
                    st.success(f"*  Reject the null hypothesis - the observed frequencies are significantly different from the expected frequencies. This suggests that there is a relationship between the two categorical variables.")
                    if p_value_1 < 0.05:
                        st.success(f"*  The p-value is less than 5%, which suggests that the relationship is statistically significant.")
                    else:
                        st.error(f"*  The p-value is greater than or equal to 5%, which suggests that the relationship is not statistically significant.")
                else:
                    st.error(f"*  Fail to reject the null hypothesis - the observed frequencies are not significantly different from the expected frequencies. This suggests that there is no relationship between the two categorical variables. The observed frequencies are not significantly different from the expected frequencies.")
                st.write("\n")
                st.write("\n")

        except TypeError:
            st.error(f'❌ Both {dependent_column_name} and {independent_column_name} columns need to be categorical/discrete with at least 2 unique values.')  
        except ValueError:
            st.error(f'❌ Both {dependent_column_name} and {independent_column_name} columns need to be categorical/discrete with at least 2 unique values.')  
        except AttributeError:
            st.error(f'❌ Both {dependent_column_name} and {independent_column_name} columns need to be categorical/discrete with at least 2 unique values.')  


	  


#The chi-square test is a statistical test used to determine whether there is a significant association between two categorical variables. To perform a chi-square test, you need to have a dataset containing two categorical variables and a target variable.

#Here's an example of the variables you might need when performing a chi-square test:

#data: A Pandas DataFrame containing the dataset.
#col1: The name of the first categorical variable.
#col2: The name of the second categorical variable.
#target_col: The name of the target variable. This is the variable that you are trying to predict based on the values of the two categorical variables.
#alpha: The significance level (also known as the alpha level) for the test. This is the probability of rejecting the null hypothesis when it is true. A common value for alpha is 0.05.
#To perform the chi-square test, you need to compute the chi-square statistic for each of the two categorical variables, as well as the p-value and the degrees of freedom for each variable. You can then compare the chi-square statistic and the p-value for each variable to the critical value and the alpha level to determine whether there is a significant association between the variable and the target variable.

#I hope this helps! Let me know if you have any questions or if 