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
 
def chi_square(df_final, file, column):

    data = df_final.select_dtypes(include=['object','int'])
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("\n")
    st.header('Chi-Square')
    with st.expander("Chi-square Test?",expanded=True):   
        st.write("A chi-square test is used to test a hypothesis regarding the distribution of a categorical variable.")
        st.markdown("- x: a categorical variable with at least two mutually exclusive and independent levels (e.g., gender: male, female) ")
        st.markdown("- y: a categorical variable with at least two mutually exclusive and independent levels (e.g., political affiliation: Democrat, Republican)")             
        st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)
    st.subheader("[👁️‍🗨️] Table Preview:")
    st.dataframe(data)

    button, chi_row, chi_col = st.columns((5,1,1), gap="small")
    rows = data.shape[0]
    cols = data.shape[1]
    with chi_row:
        st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
    with chi_col:
        st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)
    with button:
        st.write("")
        #if st.button("Download CSV"):
        #    data = data.select_dtypes(include=['object','int'])
        #    now = datetime.datetime.now()
        #    date_string = now.strftime("%Y-%m-%d")
        #    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        #    save_path = os.path.join(desktop, f'chisquare_filtered_data_csv_{date_string}.csv')
        #    data.to_csv(save_path, index=False)
        #    st.success(f'File saved successfully to {save_path}!')
    
    colored_header(
    label="",
    description="",
    color_name="violet-70",
    )  
    st.write("\n")    
    column_names = data.columns.tolist()

    # Select the columns to encode and test
    #independent_column_name  = st.selectbox("➕ Select the column name for the X (independent/CATEGORICAL/DISCRETE) variable", column_names)
    int_column_names = [col for col in column_names if data[col].dtype == 'int64']
    object_column_names = [col for col in column_names if data[col].dtype == 'object']
    filtered_column_names = int_column_names + object_column_names
    independent_column_name = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", object_column_names)

    
    st.write("\n")    
    #column_2 = st.selectbox("➕ Select the second column (categorical):", column_names)
    #dependent_column_name = st.selectbox("➕ Select the column name for the y (dependent/CATEGORICAL/DISCRETE) variable", column_names)
    dependent_column_name = column
    #st.markdown(f"<span style='color: black;'>➕ Selected the y (dependent/CATEGORICAL/DISCRETE) variable: </span> <span style='color: blue;'>{column}</span>", unsafe_allow_html=True)

    st.write("\n")    

    if independent_column_name  == dependent_column_name:
        st.error("❌ Both columns are the same. Please select different columns.")
    else:           
        try:
            if (not np.issubdtype(df_final[dependent_column_name], np.number) and df_final[dependent_column_name].nunique() < 2):
                st.error(f'❌ SELECTION ERROR #1: {dependent_column_name} column might contain categorical/string variables, column can be either discrete or continuous data with atleast 2 unique values.')
            
            elif (df_final[independent_column_name].nunique() < 2):
                st.error(f'❌ SELECTION ERROR #2: {independent_column_name} column must have atleast 2 unique values.') 
            
            elif (df_final[dependent_column_name].nunique() < 2):
                st.error(f'❌ SELECTION ERROR #3: {dependent_column_name} column must have atleast 2 unique values.') 

            elif (pd.api.types.is_string_dtype(df_final[dependent_column_name])):
                st.error(f'❌ SELECTION ERROR #4: {dependent_column_name} column might contain categorical/string variables, column must be transformed first before performing the ANOVA test.')
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
                CS1, CS2 = st.columns((1,5), gap="small")
                with CS1:
                    st.write("")
                with CS2:
                    with st.expander("Understanding the Chi-square score and Critical Value",expanded=False):   
                        st.write("A chi-square score value is a measure of the difference between the observed and expected frequencies of the outcomes of a set of events or variables. It is useful for analyzing differences in categorical variables, especially those nominal in nature.")
                        st.write("")
                        st.write("The chi-square value will assist us in determining the statistical significance of the difference between expected and observed data.")
                        st.write("")
                        st.write("Imagine you are a teacher and you want to know if your students are evenly distributed among four classes: A, B, C, and D. You count the number of students in each class and compare it to the expected number of students in each class if they were evenly distributed. The difference between the observed and expected number of students in each class is the chi-square score. If the chi-square score is high, it means that the students are not evenly distributed among the classes.")
                        st.write("")
                        st.write("Now, imagine that you want to know if the difference between the observed and expected number of students in each class is statistically significant. You can use the chi-square critical value to determine this. The chi-square critical value is a threshold for statistical significance for certain hypothesis tests and defines confidence intervals for certain parameters. If the chi-square score is greater than the chi-square critical value, then the results of the test are statistically significant.")
                
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
                    

                st.write("")
                P1, SL2 = st.columns((1,5), gap="small")
                with P1:
                        st.write("")
                with SL2:
                    with st.expander("Understanding the Significance Level and P-value",expanded=False):   
                        st.write("The p-value in ANOVA represents the probability that the differences between the groups are due to chance. A small p-value (usually less than 0.05) indicates that the differences between the groups are unlikely to be due to chance, and we can reject the null hypothesis that there is no difference between the groups. In other words, if the p-value is small, it suggests that there is a significant difference between at least two of the groups.")
                        st.write("")
                        st.write("The significance level in ANOVA works in a similar way as in other statistical tests. We set a significance level, usually at 0.05, which represents the maximum probability of making a Type I error, which is rejecting the null hypothesis when it's actually true. If the p-value is less than the significance level, we reject the null hypothesis and conclude that there is a statistically significant difference between at least two of the groups.")
                        st.write("")
                        st.write("To give an analogy, imagine you are comparing the exam scores of three different classes to see if there is a significant difference between them. The null hypothesis would be that there is no significant difference in exam scores between the three classes, and the alternative hypothesis would be that there is a significant difference between at least two of the classes.")
                        st.write("")
                        st.write("You would conduct an ANOVA test and obtain a p-value of 0.02. This means that there is a 2% chance of observing the differences in exam scores between the three classes due to chance. Since the p-value is less than the significance level of 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference in exam scores between at least two of the classes. This information could be useful in identifying areas for improvement or to make decisions about which class may need additional resources or attention.")
                        
                value_2a, value_2b = st.columns((1,5), gap="small")   
                with value_2a:
                    st.metric("P-value:",f"{p_value_1:.2f}")
                    st.metric("Significance level:",f"{100-p_value_1:.2f}")
            
                with value_2b:
                    if p_value_1 < 0.05:
                        st.success(f'* The result is statistically significant, which means that there is a less than 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                        st.success(f'* The F-value is large, which indicates that there is a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                        st.success(f'* This suggests that the {independent_column_name} has a significant effect on the {dependent_column_name}.')
                    else:
                        st.error(f'* The result is not statistically significant, which means that there is a greater than or equal to 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                        st.error(f'* The F-value is small, which indicates that there is not a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                        st.error(f'* This suggests that the {independent_column_name} has no significant effect on the {dependent_column_name}.')

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
                    st.warning(f'* The difference between the mean is greater than 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}) which suggests that there are outliers in the data.')
                else:
                    st.info(f'* The difference between the mean is less than or equal to 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}), which suggests that there are no significant outliers in the data.')

                st.write("\n")
                st.subheader("[🧪] Hypothesis Testing")
                st.write("\n")
                if p_value_1 <= 0.05:
                    result = "Reject Null Hypothesis"
                    conclusion = "There is sufficient evidence to suggest that {} is a factor on {}.".format(independent_column_name, dependent_column_name)
                    st.info("* P Value is {:.2f} which is less than or equal to 0.05 ({}); {}".format(p_value_1, result, conclusion))
                else:
                    result = "Fail to Reject Null Hypothesis"
                    conclusion = "There is not sufficient evidence to suggest that {} is a factor on {}.".format(independent_column_name, dependent_column_name)
                    st.warning("* P Value is {:.2f} which is greater than to 0.05 ({}); {}".format(p_value_1, result, conclusion))

                # Add null and alternate hypothesis statements
                null_hypothesis = "The independent variable {} has no effect on the dependent variable {}.".format(independent_column_name, dependent_column_name)
                alternate_hypothesis = "The independent variable {} has an effect on the dependent variable {}.".format(independent_column_name, dependent_column_name)
                st.write("\n\n")
                st.markdown(f"<span style='color: blue;'>Null Hypothesis (H0): </span> <span style='color: black;'>{null_hypothesis}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: blue;'>Alternate Hypothesis (H1): </span> <span style='color: black;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

                st.write("\n\n")
                st.markdown(f"<span style='color: black;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: blue;'>({independent_column_name})</span> <span style='color: black;'> has an effect on the dependent variable </span> <span style='color: blue;'>({dependent_column_name})</span>. <span style='color: black;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: blue;'>({independent_column_name})</span> <span style='color: black;'>not have an effect on the dependent variable </span> <span style='color: blue;'>({dependent_column_name})</span><span style='color: black;'>.</span>", unsafe_allow_html=True)
        

        except TypeError:
            st.error(f'❌ SELECTION ERROR #5: {dependent_column_name} column might contain categorical/string variables, column must be transformed first before performing the ANOVA test.')         
        except KeyError:
            st.error(f'❌ SELECTION ERROR #6: {dependent_column_name} column might contain categorical/string variables, column must be transformed first before performing the ANOVA test.')         

        except ValueError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
        except AttributeError:
            st.error(f'❌ Both [{dependent_column_name}] and [{independent_column_name}] columns need to be categorical/discrete with at least 2 unique values.')  
