import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header
from scipy.stats import f_oneway
from scipy.stats import f
import os
import datetime
import math
import plotly.express as px


def anova(df_final, file, column):

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header('Analysis of variance (ANOVA)')
        st.write("\n")
        
        with st.expander("One-way or Two-way ANOVA?",expanded=True):   
            st.write("The variables x and y for a one-way ANOVA are:")
            st.markdown("- x: a categorical variable with at least three levels (e.g., class A, class B, class C)")
            st.markdown("- y: a quantitative variable (e.g., height)")
            st.write("")
            st.write("The variables x and y for a two-way ANOVA are:")
            st.write("The variables x and y for a two-way ANOVA depend on whether you have replication or not. Replication means that you have multiple observations for each combination of levels of the independent variables.")
            st.write("If you have replication,")
            st.markdown("- x: a categorical variables with at least two levels (e.g., Coke, Pepsi, Sprite, Fanta)")
            st.markdown("- y: a categorical variables with at least two levels (e.g., male, female)")
            st.write("")
            st.write("If you do not have replication,")
            st.markdown("- x: a categorical variables with at least two levels (e.g., fertilizer (mixture 1, mixture 2, mixture 3))")
            st.markdown("- y: a quantitative variable (e.g., crop yield or weight of crops)")                 
            st.markdown('''
                <style>
                [data-testid="stMarkdownContainer"] ul{
                    padding-left:40px;
                }
                </style>
                ''', unsafe_allow_html=True)
        st.subheader("[👁️‍🗨️] Table Preview:")
        st.dataframe(df_final)

        button,anova_row, anova_col = st.columns((0.0001,1.5,4.5), gap="small")
        rows = df_final.shape[0]
        cols = df_final.shape[1]
        with anova_row:
            st.markdown(f"<span style='color: violet;'>➕ Filtered # of rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
        with anova_col:
            st.markdown(f"<span style='color: violet;'>➕ Filtered # of columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)

        column_names = df_final.columns.tolist()
        
        x, y = st.columns((5,4), gap="large")  
        st.sidebar.write("\n")
        independent_column_name = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", column_names)
        
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
                    st.error(f'❌ SELECTION ERROR #4: {independent_column_name} column might contain categorical/string variables, column must be transformed first before performing the ANOVA test.')
                
                elif (pd.api.types.is_string_dtype(df_final[dependent_column_name])):
                    st.error(f'❌ SELECTION ERROR #5: {dependent_column_name} column might contain categorical/string variables, column must be transformed first before performing the ANOVA test.')
                    
                #elif (pd.api.types.is_string_dtype(df_final[dependent_column_name])):
                #    st.error(f'❌ SELECTION ERROR #6: {dependent_column_name} column can only be either discrete or continuous data with atleast 2 unique values.')
                
                else:
                    st.markdown("---")    
                    st.subheader("[✍] ANOVA Test")
                    
                    ind_col_data = df_final[independent_column_name].values
                    dep_col_data = df_final[dependent_column_name].values
                    result = f_oneway(ind_col_data, dep_col_data)

                    f_value = result.statistic
                    p_value = result.pvalue

                    alpha = 0.05
                    k = len(df_final[independent_column_name].unique())
                    dfn = k-1

                    n = len(df_final)
                    dfd = n-k
                    c_value = f.ppf(1-alpha, dfn, dfd)

                    if f_value == float('inf'):
                        st.success(f'* The F-value is not defined because one of the groups has a variance of 0 or the groups have a very large difference in sample size.')
                        st.error(f'* The ANOVA test is not appropriate to use and other statistical tests should be considered.')
                    else:
                        st.write("\n")
                        value_a, value_b = st.columns((1,5), gap="small")                        
                        with value_a:
                            st.metric("F-stat:",f"{f_value:.2f}")
                            #st.write(f_value)
                            st.metric("Critical-value:",f"{c_value:.2f}")
                            #st.write(c_value)
                        with value_b:
                            if f_value > c_value:
                                st.success("The calculated F ratio is greater than the critical value, the null hypothesis is rejected, and the result is considered statistically significant. This means that there is a significant difference between the means of the groups.")
                            else:
                                st.error("The calculated F ratio is not greater than the critical value, the null hypothesis is not rejected, and the result is not considered statistically significant. This means that there is no significant difference between the means of the groups.")
                        
                        value_2a, value_2b = st.columns((1,5), gap="small")   

                        with value_2a:
                            st.metric("P-value:",f"{p_value:.2f}")
                            st.metric("Significance level:",f"{100-p_value:.2f}")
                  
                        with value_2b:
                            if p_value < 0.05:
                                st.success(f'* The result is statistically significant, which means that there is a less than 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                                st.success(f'* The F-value is large, which indicates that there is a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                                st.success(f'* This suggests that the {independent_column_name} has a significant effect on the {dependent_column_name}.')
                            else:
                                st.error(f'* The result is not statistically significant, which means that there is a greater than or equal to 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                                st.error(f'* The F-value is small, which indicates that there is not a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                                st.error(f'* This suggests that the {independent_column_name} has no significant effect on the {dependent_column_name}.')

                        x = df_final[independent_column_name]
                        y = df_final[dependent_column_name]

                        traceBox = go.Box(x=x, y=y, name=dependent_column_name)

                        dataBox = [traceBox]

                        summary_statsY = df_final[dependent_column_name].describe()

                        meanY = summary_statsY['mean']
                        medianY = summary_statsY['50%']
                        stdY = summary_statsY['std']
                        modeY = df_final[dependent_column_name].mode()

                        st.write("\n")
                        st.subheader("[📝] Descriptive Statistics for Y")
                        st.write("\n")
                        mean1a, mean2a = st.columns((1,5), gap="small")
                        with mean1a:                    
                            st.metric("Mean:",f"{meanY:.2f}")
                        with mean2a:
                            st.info("* The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")                     
                        
                        median1a, median2a = st.columns((1,5), gap="small")
                        with median1a:
                            st.metric("Median:",f"{medianY:.2f}")
                    
                        with median2a:    
                            st.info("* The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
                        
                        mode1a, mode2a = st.columns((1,5), gap="small")
                        if modeY.shape[0] == 0:
                            st.warning("This data set doesn't have a mode.")
                        else:
                            for i in range(modeY.shape[0]):
                                with mode1a:
                                    st.metric("Modes:",f"{modeY[i]:.2f}")
                                 
                                with mode2a:    
                                    st.info("* The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")

                        std1a, std2a = st.columns((1,5), gap="small")
                        with std1a:
                            st.metric("Standard Deviation:",f"{stdY:.2f}")
                           
                        with std2a:
                            st.info("* The standard deviation is a measure of how spread out the values are from the mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.")

                        st.write("\n")
                        st.subheader("[💡] Insight Statistics for Y")
                        st.write("\n")   
                     
                        if meanY > medianY:
                            st.info(f'* The mean is higher than the median, which suggests that the data is skewed to the right.')
                        elif meanY < medianY:
                            st.info(f'* The mean is lower than the median, which suggests that the data is skewed to the left.')
                        else:
                            st.info(f'* The mean is equal to the median, which suggests that the data is symmetrical.')

                        if stdY > 1:
                            st.warning(f'* The standard deviation is high (> 1), which indicates that the data is dispersed.')
                        else:
                            st.info(f'* The standard deviation is low, which indicates that the data is concentrated.')
                        if meanY > (3 * stdY):
                            st.warning(f'* The difference between the mean is greater than 3 times the standard deviation, (Mean: {meanY:.2f}, UCL:{meanY + (3 * stdY):.2f}, LCL:{meanY - (3 * stdY):.2f}) which suggests that there are outliers in the data.')
                        else:
                            st.info(f'* The difference between the mean is less than or equal to 3 times the standard deviation, (Mean: {meanY:.2f}, UCL:{meanY + (3 * stdY):.2f}, LCL:{meanY - (3 * stdY):.2f}), which suggests that there are no significant outliers in the data.')
                        
                        st.write("\n")
                        st.subheader("[🧪] Hypothesis Testing")
                        if p_value <= 0.05:
                            result = "Reject Null Hypothesis"
                            conclusion = "There is sufficient evidence to suggest that {} is a factor on {}.".format(independent_column_name, dependent_column_name)
                            st.info("* P Value is {:.2f} which is less than or equal to 0.05 ({}); {}".format(p_value, result, conclusion))
                        else:
                            result = "Fail to Reject Null Hypothesis"
                            conclusion = "There is not sufficient evidence to suggest that {} is a factor on {}.".format(independent_column_name, dependent_column_name)
                            st.warning("* P Value is {:.2f} which is greater than to 0.05 ({}); {}".format(p_value, result, conclusion))

                        null_hypothesis = "The independent variable {} has no effect on the dependent variable {}.".format(independent_column_name, dependent_column_name)
                        alternate_hypothesis = "The independent variable {} has an effect on the dependent variable {}.".format(independent_column_name, dependent_column_name)
                        st.write("\n\n")
                        st.markdown(f"<span style='color: blue;'>Null Hypothesis (H0): </span> <span style='color: black;'>{null_hypothesis}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='color: blue;'>Alternate Hypothesis (H1): </span> <span style='color: black;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

                        st.write("\n\n")
                        st.markdown(f"<span style='color: black;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: blue;'>({independent_column_name})</span> <span style='color: black;'> has an effect on the dependent variable </span> <span style='color: blue;'>({dependent_column_name})</span>. <span style='color: black;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: blue;'>({independent_column_name})</span> <span style='color: black;'>not have an effect on the dependent variable </span> <span style='color: blue;'>({dependent_column_name})</span><span style='color: black;'>.</span>", unsafe_allow_html=True)
                
                        colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )          
                        st.subheader("[📉] Graph")
                        graph1, graph2 = st.columns((4.7,7), gap="small")
                        
                        with graph2:
                        
                            fig = go.Figure()
                            fig.add_trace(go.Box(
                                y=y,
                                name="All Points",
                                jitter=0.3,
                                pointpos=-1.8,
                                boxpoints='all', # represent all points
                                marker_color='red',
                                line_color='black',
                                marker_size=3, # increase the size of the plotted points
                                line_width=2 # increase the line width
                            ))

                            fig.add_shape(
                                type='line',
                                x0=0,
                                y0=meanY,
                                x1=1,
                                y1=meanY,
                                xref='paper',
                                yref='y',
                                line=dict(
                                    color='red',
                                    width=1,
                                    dash='dash' 
                                )
                            )
                            fig.add_shape(
                                type='line',
                                x0=0,
                                y0=meanY-stdY,
                                x1=1,
                                y1=meanY-stdY,
                                xref='paper',
                                yref='y',
                                line=dict(
                                    color='green',
                                    width=1,
                                    dash='dash'  
                                )
                            )
                            fig.add_shape(
                                type='line',
                                x0=0,
                                y0=meanY+stdY,
                                x1=1,
                                y1=meanY+stdY,
                                xref='paper',
                                yref='y',
                                line=dict(
                                    color='green',
                                    width=1,
                                    dash='dash'  
                                )
                            )

                            fig.add_annotation(
                                x=1.5,
                                y=meanY,
                                text=f"Mean: {meanY:.3f}",
                                showarrow=False,
                                font=dict(
                                    color='green',
                                    size=17,
                                    family='Tahoma'
                                )
                            )
                            fig.add_annotation(
                                x=1.5,
                                y=meanY-stdY,
                                text=f"Mean - Std: {meanY-stdY:.3f}",
                                showarrow=False,
                                font=dict(
                                    color='black',
                                    size=17,
                                    family='Tahoma'
                                )
                            )
                            fig.add_annotation(
                                x=1.5,
                                y=meanY+stdY,
                                text=f"Mean + Std: {meanY+stdY:.3f}",
                                showarrow=False,
                                font=dict(
                                    color='black',
                                    size=17,
                                    family='Tahoma'
                                )
                            )            
                                        
                            fig.update_layout(
                                title_text=f"Box Plot Styling Outliers: {dependent_column_name}",
                                yaxis_title=f"{dependent_column_name}",
                                xaxis_title=f"{independent_column_name}"
                                )

                            st.plotly_chart(fig)     
              
                        with graph1:
                            st.write("#")
                            #st.write("The box plot provides an overview of the distribution of the dependent variable, `" + dependent_column_name + "`, across different values of the independent variable, `" + independent_column_name + "`. The x-axis of the chart represents the independent variable, and the y-axis represents the dependent variable. The box represents the interquartile range (IQR) of the data, which is the range between the first quartile (25th percentile) and the third quartile (75th percentile). The horizontal line within the box is the median (50th percentile). The whiskers represent the range of the data, excluding outliers. Outliers are represented as individual dots.")
                            st.write("A box plot, shows how a set of data is distributed visually. To visualize the distribution of numerical data and spot trends, patterns, and outliers, statistical analysis frequently uses this technique.")
                            st.write("The box represents range between the first quartile (25th percentile) and the third quartile, or the interquartile range (IQR), is shown by the box in the box plot (75th percentile). The first quartile is the percentage that divides the lowest 25% of the data from the remaining data, and the third quartile, the highest 25% of the data.")
                            st.write("The median, which is the number separating the lowest 50% from the highest 50% of the data, is the horizontal line inside the box. It gives the data a measure of central tendency and serves as a reliable guide to the typical value in the data set.")
        
                        st.write("The whiskers extend from the box to represent the range of the data, excluding outliers. Outliers are values that are significantly different from the rest of the data and are represented as individual dots outside the whiskers. They can represent errors in data collection, measurement, or data entry and are important to identify as they can skew the results of a statistical analysis.")
                        st.markdown(f"<span style='color: black;'>The x-axis of the box plot represents the independent variable </span><span style='color: blue;'>({independent_column_name})</span><span style='color: black;'> which is the variable that you want to examine the relationship between it and the dependent variable. The y-axis represents the dependent variable<span style='color: blue;'>({dependent_column_name})</span><span style='color: black;'> which is the variable you want to analyze its distribution.</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='color: black;'>In conclusion, the box plot provides a quick and effective way to visualize the distribution of the dependent variable </span><span style='color: blue;'>({dependent_column_name})</span><span style='color: black;'> across different values of the independent variable</span><span style='color: blue;'>({independent_column_name})</span><span style='color: black;'> . By examining the heights and ranges of the boxes, you can gain insights into the relationship between the dependent variable and the independent variable and identify trends, patterns, and outliers in the data.</span>", unsafe_allow_html=True)
