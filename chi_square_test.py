import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py

from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency, chi2
from streamlit_extras.colored_header import colored_header

#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import os
import datetime

#def chi_square_fixed(fixed_data, col1, col2):
def chi_square_fixed(fixed_data, file):
    #py.init_notebook_mode()

    # Select only categorical columns
    fixed_data = fixed_data.select_dtypes(include=['object','int'])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("\n")
    st.header('Chi-Square')
    # Read in the file
    st.subheader("[👁️‍🗨️] Table Preview:")
    st.dataframe(fixed_data)
    #ag_grid = AgGrid(
    #fixed_data,
    #key='unique_key_1',
    #height=300, 
    #width='100%',
    #data_return_mode=DataReturnMode.FILTERED_AND_SORTED, 
    #update_mode=GridUpdateMode.FILTERING_CHANGED,
    #fit_columns_on_grid_load=True
    ##allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
    #)
    chi_row, chi_col = st.columns((1,4), gap="small")
    rows = fixed_data.shape[0]
    cols = fixed_data.shape[1]
    with chi_row:
        st.markdown(f"<span style='color: blue;'>Current number of rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
    with chi_col:
        st.markdown(f"<span style='color: blue;'>Current number of columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)

    if st.button("Download CSV"):
        # Select only numeric columns
        #ag_grid = fixed_data.select_dtypes(include=['float'])
        fixed_data = fixed_data.select_dtypes(include=['object','int'])
        # Get current date
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        # Set default save location to desktop
        desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        save_path = os.path.join(desktop, f'chisquare_filtered_data_csv_{date_string}.csv')
        # write data to the selected file
        fixed_data.to_csv(save_path, index=False)
        st.success(f'File saved successfully to {save_path}!')

    colored_header(
    label="",
    description="",
    color_name="violet-70",
    )  

    #st.write(f"Number of rows :{fixed_data.shape[0]:.0f}")
    #st.write(f"Number of columns :{fixed_data.shape[1]:.0f}")    
    # Get the list of column names
    column_names = fixed_data.columns.tolist()
    st.write("\n")   
    # Select the columns to encode and test
    st.write('Dataframe fixed.')
    independent_column_name  = st.selectbox("➕ Select the column name for the X (independent/CATEGORICAL/DISCRETE) variable", column_names)
    st.write("\n")   
    #column_2 = st.selectbox("➕ Select the second column (categorical):", column_names)
    dependent_column_name = st.selectbox("➕ Select the column name for the y (dependent/CATEGORICAL/DISCRETE) variable", column_names)
    st.write("\n")   
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

                # Calculate the mean of Y
                mean_y = np.mean(fixed_data[dependent_column_name])

                # Calculate the median of Y
                median_y = np.median(fixed_data[dependent_column_name])

                # Calculate the mode of Y
                mode_y = fixed_data[dependent_column_name].mode()[0]

                # Calculate the standard deviation of Y
                std_y = np.std(fixed_data[dependent_column_name])

                # Display the results
                st.subheader("[✍] Chi-Square Test")
                st.write("\n")
                chi_cola1, chi_colb2, = st.columns((1,5), gap="small")
                with chi_cola1:
                    #st.write(f'Chi-Square Score: {chi2_score_1:.2f}',  prefix="\t")
                    st.metric("Chi-Square Score:",f"{chi2_score_1:.2f}")
                with chi_cola1:
                    #st.write(f'Critical Value: {critical_value_1:.2f}', prefix="\t")
                    st.metric("Critical Value:",f"{critical_value_1:.2f}")
                with chi_colb2:            
                    if chi2_score_1 > critical_value_1:
                        st.success(f"*  Reject the null hypothesis - the observed frequencies are significantly different from the expected frequencies. This suggests that there is a relationship between the two categorical variables.")
                    else:
                        st.error(f"*  Fail to reject the null hypothesis - the observed frequencies are not significantly different from the expected frequencies. This suggests that there is no relationship between the two categorical variables. The observed frequencies are not significantly different from the expected frequencies.")
                    

                pval1, pval2, = st.columns((1,5), gap="small")
                with pval1:
                    #st.write(f'P-Value: {p_value_1:.2f}', prefix="\t")
                    st.metric("P-Value:",f"{p_value_1:.2f}")
                    #st.write(f'Degrees of Freedom: {degrees_of_freedom_1:.2f}', prefix="\t")
                with pval2:

                    if p_value_1 < 0.05:
                        st.success(f"*  The p-value is less than 0.05, which suggests that the relationship is statistically significant.")
                    else:
                        st.error(f"*  The p-value is greater than or equal to 0.05, which suggests that the relationship is not statistically significant.")

                st.write("\n")
                st.subheader("[📝] Summary Statistics for Y")
                st.write("\n")
                #st.write(f'Mean: {mean_y:.2f}')
                mean1a, mean2a = st.columns((1,5), gap="small")
                with mean1a:
                    st.metric("Mean:",f"{mean_y:.2f}")
                with mean2a:
                    st.info("* The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")
                median1a, median2a = st.columns((1,5), gap="small")
                with median1a:
                    st.metric("Median:",f"{median_y:.2f}")
                with median2a:
                    st.info("* The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
                mode1a, mode2a = st.columns((1,5), gap="small")
                with mode1a:
                    st.metric("Mode:",f"{mode_y:.2f}")
                with mode2a:    
                    st.info("* The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")
                std1a, std2a = st.columns((1,5), gap="small")
                with std1a:
                    st.metric("Standard Deviation:",f"{std_y:.2f}")
                with std2a:
                    st.info("* The standard deviation is a measure of how spread out the values are from the mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.")

                #st.write("When you have the mean, median, mode and standard deviation of your data, it gives you an idea of the distribution of your data. Knowing these values can help you identify patterns in your data, and make decisions on how to analyze or interpret the data.")
                #st.write("If the mean, median, mode and standard deviation are close to each other, it suggests that your data is not skewed, but if the values are far from each other, it suggests that your data is skewed.")
                #st.write("If the mean and median are very close to each other, but the standard deviation is high, this suggests that most of the data is clustered around the mean, but there are a few outliers that are far away from the mean.")
                #st.write("If the mean and median are far from each other, and the standard deviation is high, it suggests that the data is skewed and the distribution is not normal.")
                st.write("\n")
                st.subheader("[💡] Insight Statistics for Y")
                st.write("\n")
                if mean_y > median_y:
                    st.info(f'* The mean is higher than the median, which suggests that the data is skewed to the right.')
                elif mean_y < median_y:
                    st.info(f'* The mean is lower than the median, which suggests that the data is skewed to the left.')
                else:
                    st.info(f'* The mean is equal to the median, which suggests that the data is symmetrical.')

                if std_y > 1:
                    st.warning(f'* The standard deviation is high (> 1), which indicates that the data is dispersed.')
                else:
                    st.info(f'* The standard deviation is low, which indicates that the data is concentrated.')

                if mean_y > (3 * std_y):
                    #st.warning(f'* The difference between the mean and median is greater than 3 times the standard deviation, which suggests that there are outliers in the data.')
                    st.warning(f'* The difference between the mean is greater than 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}) which suggests that there are outliers in the data.')
                else:
                    #st.info(f'* The difference between the mean and median is less than or equal to 3 times the standard deviation, which suggests that there are no significant outliers in the data.')
                    st.info(f'* The difference between the mean is less than or equal to 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}), which suggests that there are no significant outliers in the data.')

        except TypeError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
        except ValueError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
        except AttributeError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
 


def chi_square(data, file):

    # Select only categorical columns
    data = data.select_dtypes(include=['object','int'])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("\n")
    st.header('Chi-Square')
    # Read in the file
    st.subheader("[👁️‍🗨️] Table Preview:")
    st.dataframe(data)
    #ag_grid = AgGrid(
    #data,
    #key='unique_key_2',
    #height=300, 
    #width='100%',
    #data_return_mode=DataReturnMode.FILTERED_AND_SORTED, 
    #update_mode=GridUpdateMode.FILTERING_CHANGED,
    #fit_columns_on_grid_load=True
    ##allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
    #)
    button, chi_row, chi_col = st.columns((5,1,1), gap="small")
    rows = data.shape[0]
    cols = data.shape[1]
    with chi_row:
        st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
    with chi_col:
        st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)
    with button:
        if st.button("Download CSV"):
            # Select only numeric columns
            #ag_grid = data.select_dtypes(include=['float'])
            data = data.select_dtypes(include=['object','int'])
            # Get current date
            now = datetime.datetime.now()
            date_string = now.strftime("%Y-%m-%d")
            # Set default save location to desktop
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            save_path = os.path.join(desktop, f'chisquare_filtered_data_csv_{date_string}.csv')
            # write data to the selected file
            data.to_csv(save_path, index=False)
            st.success(f'File saved successfully to {save_path}!')
    
    colored_header(
    label="",
    description="",
    color_name="violet-70",
    )  
    st.write("\n")    
    #st.write(f"Number of rows :{data.shape[0]:.0f}")
    #st.write(f"Number of columns :{data.shape[1]:.0f}")       
    # Get the list of column names
    column_names = data.columns.tolist()

    # Select the columns to encode and test
    independent_column_name  = st.selectbox("➕ Select the column name for the X (independent/CATEGORICAL/DISCRETE) variable", column_names)
    st.write("\n")    
    #column_2 = st.selectbox("➕ Select the second column (categorical):", column_names)
    dependent_column_name = st.selectbox("➕ Select the column name for the y (dependent/CATEGORICAL/DISCRETE) variable", column_names)
    st.write("\n")    
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

                # Calculate the mean of Y
                mean_y = np.mean(data[dependent_column_name])

                # Calculate the median of Y
                median_y = np.median(data[dependent_column_name])

                # Calculate the mode of Y
                mode_y = data[dependent_column_name].mode()[0]

                # Calculate the standard deviation of Y
                std_y = np.std(data[dependent_column_name])

                # Display the results
                st.subheader("[✍] Chi-Square Test")
                st.write("\n")
                chi_cola1, chi_colb2, = st.columns((1,5), gap="small")
                with chi_cola1:
                    #st.write(f'Chi-Square Score: {chi2_score_1:.2f}',  prefix="\t")
                    st.metric("Chi-Square Score:",f"{chi2_score_1:.2f}")
                with chi_cola1:
                    #st.write(f'Critical Value: {critical_value_1:.2f}', prefix="\t")
                    st.metric("Critical Value:",f"{critical_value_1:.2f}")
                with chi_colb2:            
                    if chi2_score_1 > critical_value_1:
                        st.success(f"*  Reject the null hypothesis - the observed frequencies are significantly different from the expected frequencies. This suggests that there is a relationship between the two categorical variables.")
                    else:
                        st.error(f"*  Fail to reject the null hypothesis - the observed frequencies are not significantly different from the expected frequencies. This suggests that there is no relationship between the two categorical variables. The observed frequencies are not significantly different from the expected frequencies.")
                    

                pval1, pval2, = st.columns((1,5), gap="small")
                with pval1:
                    #st.write(f'P-Value: {p_value_1:.2f}', prefix="\t")
                    st.metric("P-Value:",f"{p_value_1:.2f}")
                    #st.write(f'Degrees of Freedom: {degrees_of_freedom_1:.2f}', prefix="\t")
                with pval2:

                    if p_value_1 < 0.05:
                        st.success(f"*  The p-value is less than 0.05, which suggests that the relationship is statistically significant.")
                    else:
                        st.error(f"*  The p-value is greater than or equal to 0.05, which suggests that the relationship is not statistically significant.")

                st.write("\n")
                st.subheader("[📝] Descriptive Statistics for Y")
                st.write("\n")
                #st.write(f'Mean: {mean_y:.2f}')
                mean1a, mean2a = st.columns((1,5), gap="small")
                with mean1a:
                    st.metric("Mean:",f"{mean_y:.2f}")
                with mean2a:
                    st.info("* The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")
                median1a, median2a = st.columns((1,5), gap="small")
                with median1a:
                    st.metric("Median:",f"{median_y:.2f}")
                with median2a:
                    st.info("* The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
                mode1a, mode2a = st.columns((1,5), gap="small")
                with mode1a:
                    st.metric("Mode:",f"{mode_y:.2f}")
                with mode2a:    
                    st.info("* The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")
                std1a, std2a = st.columns((1,5), gap="small")
                with std1a:
                    st.metric("Standard Deviation:",f"{std_y:.2f}")
                with std2a:
                    st.info("* The standard deviation is a measure of how spread out the values are from the mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.")

                #st.write("When you have the mean, median, mode and standard deviation of your data, it gives you an idea of the distribution of your data. Knowing these values can help you identify patterns in your data, and make decisions on how to analyze or interpret the data.")
                #st.write("If the mean, median, mode and standard deviation are close to each other, it suggests that your data is not skewed, but if the values are far from each other, it suggests that your data is skewed.")
                #st.write("If the mean and median are very close to each other, but the standard deviation is high, this suggests that most of the data is clustered around the mean, but there are a few outliers that are far away from the mean.")
                #st.write("If the mean and median are far from each other, and the standard deviation is high, it suggests that the data is skewed and the distribution is not normal.")
                st.write("\n")
                st.subheader("[💡] Insight Statistics for Y")
                st.write("\n")
                if mean_y > median_y:
                    st.info(f'* The mean is higher than the median, which suggests that the data is skewed to the right.')
                elif mean_y < median_y:
                    st.info(f'* The mean is lower than the median, which suggests that the data is skewed to the left.')
                else:
                    st.info(f'* The mean is equal to the median, which suggests that the data is symmetrical.')

                if std_y > 1:
                    st.warning(f'* The standard deviation is high (> 1), which indicates that the data is dispersed.')
                else:
                    st.info(f'* The standard deviation is low, which indicates that the data is concentrated.')

                if mean_y > (3 * std_y):
                    #st.warning(f'* The difference between the mean and median is greater than 3 times the standard deviation, which suggests that there are outliers in the data.')
                    st.warning(f'* The difference between the mean is greater than 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}) which suggests that there are outliers in the data.')
                else:
                    #st.info(f'* The difference between the mean and median is less than or equal to 3 times the standard deviation, which suggests that there are no significant outliers in the data.')
                    st.info(f'* The difference between the mean is less than or equal to 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}), which suggests that there are no significant outliers in the data.')

                st.write("\n")
                st.write("\n")

        except TypeError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
        except ValueError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
        except AttributeError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  


