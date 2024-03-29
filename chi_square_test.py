import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py

from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency, chi2
from streamlit_extras.colored_header import colored_header

import os
import datetime

from functions import transform_column, transformation_check, is_ordinal

 
def chi_square(df_final, file, column, test):

    data = df_final.select_dtypes(include=['object','int64'])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header('Chi-Square')
    with st.expander("What is Chi-square Test?",expanded=True):   
        st.write("A chi-square test is used to test a hypothesis regarding the distribution of a categorical variable.")
        st.markdown("- The variables x and y can be any categorical variable with at least two mutually exclusive and independent levels (e.g., color: red, green, blue or gender: male, female).")    
        st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)

        #if st.button("Download CSV"):
        #    data = data.select_dtypes(include=['object','int'])
        #    now = datetime.datetime.now()
        #    date_string = now.strftime("%Y-%m-%d")
        #    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        #    save_path = os.path.join(desktop, f'chisquare_filtered_data_csv_{date_string}.csv')
        #    data.to_csv(save_path, index=False)
        #    st.success(f'File saved successfully to {save_path}!')
  
    column_names = df_final.columns.tolist()

    int_column_names = [col for col in column_names if df_final[col].dtype == 'int64']
    object_column_names = [col for col in column_names if df_final[col].dtype == 'object']
    filtered_column_names = int_column_names + object_column_names
    independent_column_name = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", [""] + object_column_names)

    dependent_column_name = column

    if independent_column_name  == dependent_column_name:
        st.error("❌ Both columns are the same. Please select different columns.")
    else:           
        try:
            if independent_column_name == "":
                st.write("")
            if (not np.issubdtype(df_final[dependent_column_name], np.number) and df_final[dependent_column_name].nunique() < 2):
                st.error(f'❌ SELECTION ERROR #1: {dependent_column_name} column might contain categorical/string variables, column can be either discrete or continuous data with atleast 2 unique values.')
            
            elif (df_final[independent_column_name].nunique() < 2):
                st.error(f'❌ SELECTION ERROR #2: {independent_column_name} column must have atleast 2 unique values.') 
            
            elif (df_final[dependent_column_name].nunique() < 2):
                st.error(f'❌ SELECTION ERROR #3: {dependent_column_name} column must have atleast 2 unique values.') 

            elif (pd.api.types.is_string_dtype(df_final[dependent_column_name])):
                st.error(f'❌ SELECTION ERROR #4: {dependent_column_name} column might contain categorical/string variables, column must be transformed first before performing the Chi-square test.')
            else:
                isNumerical = []
                for columndata in df_final.columns:
                    if df_final[columndata].dtype == np.int64 or df_final[columndata].dtype == np.float64:
                        isNumerical.append("Quantitative")
                    else:
                        isNumerical.append("Categorical")

                #transformation_check(df_final, isNumerical, column, test)
            
                # Select the significance level
                #alpha = st.slider("Select the significance level (alpha):", 0.0, 1.0, 0.05)
                alpha = 0.05

                # Initialize the encoder
                # encoder = OrdinalEncoder()

                # Encode the selected columns
                # fixed_data[[independent_column_name , column_2]] = encoder.fit_transform(fixed_data[[independent_column_name , column_2]])
                # df_final[[independent_column_name ]] = encoder.fit_transform(df_final[[independent_column_name ]])
                
                # Identify the data type of x_col column
                x_col_type = None
                if data[independent_column_name].dtype == np.int64:
                    x_col_type = "integer"
                elif data[independent_column_name].dtype == np.float64:
                    x_col_type = "float"
                else:
                    x_col_type = "object"

                levels = {}
                if x_col_type == "integer":
                    unique_values = data[independent_column_name].nunique()
                    if unique_values == 2:
                        levels[independent_column_name] = "binary"
                    else:
                        levels[independent_column_name] = "discrete"
                elif x_col_type == "float":
                    unique_values = data[independent_column_name].nunique()
                    if unique_values == 2:
                        levels[independent_column_name] = "binary"
                    else:
                        levels[independent_column_name] = "continuous"
                else:
                    unique_values = data[independent_column_name].nunique()
                    if unique_values == 2:
                        levels[independent_column_name] = "binary"
                    else:
                        if is_ordinal(data[independent_column_name]):
                            levels[independent_column_name] = "ordinal"
                        else:
                            levels[independent_column_name] = "nominal"

                if levels[independent_column_name] == "nominal" and unique_values > 2:
                    recommended_method = "One-Hot"
                elif levels[independent_column_name] == "ordinal":
                    recommended_method = "Ordinal"
                elif levels[independent_column_name] == "binary":
                    recommended_method = "Label"
                else:
                    data[independent_column_name] = data[independent_column_name].values.reshape(-1, 1) # Convert to 2D array
                    
                chi2_score_1, p_value_1, _, _ = chi2_contingency(pd.crosstab(df_final[independent_column_name], df_final[dependent_column_name]), correction=False)

                # Compute the degrees of freedom and the critical value
                degrees_of_freedom_1 = df_final[independent_column_name ].nunique() - 1
                #degrees_of_freedom_2 = fixed_data[column_2].nunique() - 1
                critical_value_1 = chi2.ppf(q=1-alpha, df=degrees_of_freedom_1)
                #critical_value_2 = chi2.ppf(q=1-alpha, df=degrees_of_freedom_2)

                mean_y = np.mean(df_final[dependent_column_name])

                median_y = np.median(df_final[dependent_column_name])

                mode_y = df_final[dependent_column_name].mode()[0]

                std_y = np.std(df_final[dependent_column_name])

                st.subheader("Table Preview:")
                st.dataframe(df_final, height = 400)

                button, chi_row, chi_col = st.columns((0.0001,1.5,4.5), gap="small")
                rows = df_final.shape[0]
                cols = df_final.shape[1]
                with chi_row:
                    st.markdown(f"<span style='color: #803df5;'>➕ Number of rows : </span> <span style='color: #803df5;'>{rows}</span>", unsafe_allow_html=True)
                with chi_col:
                    st.markdown(f"<span style='color: #803df5;'>➕ Number of columns : </span> <span style='color: #803df5;'>{cols}</span>", unsafe_allow_html=True)
                with button:
                    st.write("")
                colored_header(
                label="",
                description="",
                color_name="violet-70",
                )  
                st.subheader("Chi-Square Test")

                st.write("\n")
   
                with st.expander("Understanding the Significance Level and P-value",expanded=False):   
                    st.write("The p-value represents the probability that the differences between the groups are due to chance. A small p-value (usually less than 0.05) indicates that the differences between the groups are unlikely to be due to chance, and we can reject the null hypothesis that there is no difference between the groups. In other words, if the p-value is small, it suggests that there is a significant difference between at least two of the groups.")
                    st.write("")
                    st.write("The significance level works in a similar way as in other statistical tests. We set a significance level, usually at 0.05, which represents the maximum probability of making a Type I error, which is rejecting the null hypothesis when it's actually true. If the p-value is less than the significance level, we reject the null hypothesis and conclude that there is a statistically significant difference between at least two of the groups.")
                    st.write("")
                    st.write("To give an analogy, imagine you are comparing the exam scores of three different classes to see if there is a significant difference between them. The null hypothesis would be that there is no significant difference in exam scores between the three classes, and the alternative hypothesis would be that there is a significant difference between at least two of the classes.")
                    st.write("")
                    st.write("You would conduct an ANOVA test and obtain a p-value of 0.02. This means that there is a 2% chance of observing the differences in exam scores between the three classes due to chance. Since the p-value is less than the significance level of 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference in exam scores between at least two of the classes. This information could be useful in identifying areas for improvement or to make decisions about which class may need additional resources or attention.")
                    
                pvalue1a, pvalue2a = st.columns((2,5), gap="small")
                with pvalue1a:
                    st.metric("P-value",f"{p_value_1:.3f}")       

                with pvalue2a:    
                    st.metric("Significance level:",f"{100-p_value_1:.3f}")

                if p_value_1 < 0.05:
                    st.success(f' With p-value of {p_value_1:.3f} that is less than 0.05, it means that the results for the relationship between {independent_column_name} and {dependent_column_name} are statistically significant.')
                    st.write("There is a less than 5% chance that the observed difference between the groups happened due to natural differences alone. This provides strong evidence that the independent variable has a significant impact on the dependent variable.")
                else:
                    st.warning(f' With p-value of {p_value_1:.3f} that is greater than 0.05, it means that the results for the relationship between {independent_column_name} and {dependent_column_name} are not statistically significant.')
                    st.write("There is a greater than or equal to 5% chance that the observed difference between the groups happened due to natural differences alone. This suggests that the independent variable does not have a significant impact on the dependent variable.")   

                st.write("\n")
                st.subheader("Hypothesis Testing")
                st.write("\n")
                if p_value_1 <= 0.05:
                    result = "Reject Null Hypothesis"
                    conclusion = "There is sufficient evidence to suggest that {} is a factor on {}.".format(independent_column_name, dependent_column_name)
                    st.success("P Value is {:.3f} which is less than or equal to 0.05 ({}); {}".format(p_value_1, result, conclusion))
                else:
                    result = "Fail to Reject Null Hypothesis"
                    conclusion = "There is not sufficient evidence to suggest that {} is a factor on {}.".format(independent_column_name, dependent_column_name)
                    st.warning("P Value is {:.3f} which is greater than to 0.05 ({}); {}".format(p_value_1, result, conclusion))

                null_hypothesis = "The independent variable {} has no effect on the dependent variable {}.".format(independent_column_name, dependent_column_name)
                alternate_hypothesis = "The independent variable {} has an effect on the dependent variable {}.".format(independent_column_name, dependent_column_name)
                st.write("\n\n")
                st.markdown(f"<span style='color: #803df5; font-weight: bold;'>Null Hypothesis (H0): </span> <span style='color: #344a80;'>{null_hypothesis}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: #803df5; font-weight: bold;'>Alternate Hypothesis (H1): </span> <span style='color: #344a80;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

                st.write("\n\n")
                st.markdown(f"<span style='color: #344a80;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: #803df5;'>({independent_column_name})</span> <span style='color: #344a80;'> has an effect on the dependent variable </span> <span style='color: #803df5;'>({dependent_column_name})</span>. <span style='color: #344a80;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: #803df5;'>({independent_column_name})</span> <span style='color: #344a80;'>not have an effect on the dependent variable </span> <span style='color: #803df5;'>({dependent_column_name})</span><span style='color: #344a80;'>.</span>", unsafe_allow_html=True)

                colored_header(
                label="",
                description="",
                color_name="violet-70",
                )  


                # st.write("\n")
                # st.subheader(f"Descriptive Statistics for 'Y' ({dependent_column_name})")
                # st.write("\n")
                # mean, median, mode, std_dev = st.columns((2.5,2.5,2.5,2.5), gap="small")
                # with mean:
                #     st.metric("Mean:",f"{mean_y:.3f}")
                # with median:
                #     st.metric("Median:",f"{median_y:.3f}")
                # with mode:
                #     st.metric("Mode:",f"{mode_y:.3f}")
                # with std_dev:
                #     st.metric("Standard Deviation:",f"{std_y:.3f}")
                
                # meanS, medianS, modeS, std_devS = st.columns((2.5,2.5,2.5,2.5), gap="small")
                # with meanS:
                #     st.info(" The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")
                # with medianS:
                #     st.info(" The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
                # with modeS:
                #     st.info(" The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")
                # with std_devS:
                #     st.info(" The standard deviation is a measure of how spread out the values are from the mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.")

                # st.write("\n")
                # st.subheader(f"Insight Statistics for 'Y' ({dependent_column_name})")
                # st.write("\n")
                # if mean_y > median_y:
                #     st.write(f' The mean is higher than the median, which suggests that the data is skewed to the right.')
                # elif mean_y > median_y:
                #     st.write(f' The mean is lower than the median, which suggests that the data is skewed to the left.')
                # else:
                #     st.write(f' The mean is equal to the median, which suggests that the data is symmetrical.')

                # if std_y > 1:
                #     st.markdown(
                #         f"<span style='color: #344a80;'> The standard deviation is low , </span> "
                #         f"<span style='color: red;'>(>1)</span> "
                #         f"<span style='color: #344a80;'>, which indicates that the data is concentrated. </span>",
                #         unsafe_allow_html=True
                #     )
                # else:
                #     st.markdown(
                #         f"<span style='color: #344a80;'> The standard deviation is low , </span> "
                #         f"<span style='color: #803df5;'>(<=1)</span> "
                #         f"<span style='color: #344a80;'>, which indicates that the data is concentrated. </span>",
                #         unsafe_allow_html=True
                #     )

                # if mean_y > (3 * std_y):
                #     st.markdown(
                #         f"<span style='color: #344a80;'> The difference between the mean is greater than 3 times the standard deviation, </span> "
                #         f"<span style='color: red;'>(Mean: {mean_y:.3f}, UCL:{mean_y + (3 * std_y):.3f}, LCL:{mean_y - (3 * std_y):.3f})</span> "
                #         f"<span style='color: #344a80;'>, which suggests that there might be significant outliers in the data. </span>",
                #         unsafe_allow_html=True
                #     )
                # else:
                #     st.markdown(
                #         f"<span style='color: #344a80;'> The difference between the mean is less than or equal to 3 times the standard deviation, </span> "
                #         f"<span style='color: #803df5;'>(Mean: {mean_y:.3f}, UCL:{mean_y + (3 * std_y):.3f}, LCL:{mean_y - (3 * std_y):.3f})</span> "
                #         f"<span style='color: #344a80;'>, which suggests that the data falls within the expected range based on control limits. </span>",
                #         unsafe_allow_html=True
                #     )

        except (TypeError,KeyError):
           #  st.error(f'❌ SELECTION ERROR #5: {dependent_column_name} column might contain categorical/string variables, column must be transformed first before performing the Chi-square test.')         
            st.write("")
        except (ValueError,AttributeError):
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
