import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header
from scipy.stats import f_oneway
from scipy.stats import f
import os
import datetime
import math


def anova(data, file):
    
        data = data.select_dtypes(include=['object','float','int'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header('Analysis of variance (ANOVA)')
        st.write("\n")
        st.subheader("[👁️‍🗨️] Table Preview:")
        st.dataframe(data)

        anova_row, anova_col = st.columns((1,5), gap="small")
        rows = data.shape[0]
        cols = data.shape[1]
        with anova_row:
            st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
        with anova_col:
            st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)



        colored_header(
        label="",
        description="",
        color_name="violet-70",
        )  
        st.write("\n")
        column_names = data.columns.tolist()
        independent_column_name = st.selectbox('➕ Select the column name for the X (independent/DISCRETE) variable:', column_names)
        st.write("\n")
        dependent_column_name = st.selectbox('➕ Select the column name for the y (dependent/CONTINUOUS) variable:', column_names)
        st.write("\n")
    
        if independent_column_name == dependent_column_name:
            st.error("❌ Both columns are the same. Please select different columns.")
        else:           
            try:
                if ((not pd.api.types.is_integer_dtype(data[independent_column_name])) and data[independent_column_name].nunique() < 2):
                    st.error(f'❌ {independent_column_name} column must be either discrete data with atleast 2 unique values.')
                
                elif (not pd.api.types.is_float_dtype(data[dependent_column_name]) and not np.issubdtype(data[dependent_column_name], np.number)):
                    st.error(f'❌ {dependent_column_name} column must be a continuous variable.') 
                
                elif (pd.api.types.is_string_dtype(data[independent_column_name]) and pd.api.types.is_float_dtype(data[independent_column_name]) and np.issubdtype(data[independent_column_name], np.number)):
                    st.error(f'❌ {independent_column_name} column must be either discrete data with atleast 2 unique values.')
                
                elif ((pd.api.types.is_string_dtype(data[dependent_column_name]) or pd.api.types.is_integer_dtype(data[dependent_column_name])) or not data[independent_column_name].nunique() > 1):
                    st.error(f'❌ {dependent_column_name} column must be a continuous variable.') 
                
                else:
                    colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )          
                    st.subheader("[✍] ANOVA Test")
                    
                    ind_col_data = data[independent_column_name].values
                    dep_col_data = data[dependent_column_name].values
                    result = f_oneway(ind_col_data, dep_col_data)
            
              
                    f_value = result.statistic
                 
                    p_value = result.pvalue

                    alpha = 0.05
                   
                    k = len(data[independent_column_name].unique())
                    dfn = k-1

                    n = len(data)
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
                            st.metric("Critical-value:",f"{c_value:.2f}")
                        with value_b:
                            if f_value > c_value:
                                st.success("The calculated F ratio is greater than the critical value, the null hypothesis is rejected, and the result is considered statistically significant. This means that there is a significant difference between the means of the groups.")
                            else:
                                st.error("The calculated F ratio is not greater than the critical value, the null hypothesis is not rejected, and the result is not considered statistically significant. This means that there is no significant difference between the means of the groups.")
                        
                        value_2a, value_2b = st.columns((1,5), gap="small")   
      
                        with value_2a:
                            st.metric("P-value (significance level):",f"{p_value:.2f}")
                        with value_2b:
                            if p_value < 0.05:
                                st.success(f'* The result is statistically significant, which means that there is a less than 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                                st.success(f'* The F-value is large, which indicates that there is a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                                st.success(f'* This suggests that the {independent_column_name} has a significant effect on the {dependent_column_name}.')
                            else:
                                st.error(f'* The result is not statistically significant, which means that there is a greater than or equal to 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                                st.error(f'* The F-value is small, which indicates that there is not a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                                st.error(f'* This suggests that the {independent_column_name} has no significant effect on the {dependent_column_name}.')

                        x = data[independent_column_name]
                        y = data[dependent_column_name]

                        traceBox = go.Box(x=x, y=y, name=dependent_column_name)

                        dataBox = [traceBox]

                        summary_statsY = data[dependent_column_name].describe()

                        meanY = summary_statsY['mean']
                        medianY = summary_statsY['50%']
                        stdY = summary_statsY['std']
                        modeY = data[dependent_column_name].mode()  
                       
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
                        # Analyze the results
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
                        if p_value <= 0.05:
                            result = "Reject Null Hypothesis"
                            conclusion = "{} is a factor on {}.".format(independent_column_name, dependent_column_name)
                            st.info("* P Value is {:.2f} which is less than or equal to 0.05 ({}); {}".format(p_value, result, conclusion))
                        else:
                            result = "Fail to Reject Null Hypothesis"
                            conclusion = "{} is not a factor on {}.".format(independent_column_name, dependent_column_name)
                            st.warning("* P Value is {:.2f} which is greater than to 0.05 ({}); {}".format(p_value, result, conclusion))

                        
                        
                        st.write("\n")
                        colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )          
                        st.subheader("[📉] Graph")
                        #st.write(f'➕ Box Plot: {dependent_column_name} (outcome/dependent/output) ')
                        # Select the data for the box plot
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=y,
                            name="All Points",
                            jitter=0.3,
                            pointpos=-1.8,
                            boxpoints='all', # represent all points
                            marker_color='skyblue',
                            line_color='rgb(7,40,89)',
                            marker_size=3, # increase the size of the plotted points
                            line_width=2 # increase the line width
                        ))

                        fig.add_trace(go.Box(
                            y=y,
                            name="Only Whiskers",
                            boxpoints=False, # no data points
                            marker_color='rgb(9,56,125)',
                            line_color='rgb(9,56,125)',
                            marker_size=10, # increase the size of the plotted points
                            line_width=2 # increase the line width
                        ))

                        fig.add_trace(go.Box(
                            y=y,
                            name="Suspected Outliers",
                            boxpoints='suspectedoutliers', # only suspected outliers
                            marker=dict(
                                color='rgb(8,81,156)',
                                outliercolor='rgba(219, 64, 82, 0.6)',
                                line=dict(
                                    outliercolor='rgba(219, 64, 82, 0.6)',
                                    outlierwidth=2)),
                            line_color='rgb(8,81,156)',
                            marker_size=10, # increase the size of the plotted points
                            line_width=2 # increase the line width
                        ))

                        fig.add_trace(go.Box(
                            y=y,
                            name="Whiskers and Outliers",
                            boxpoints='outliers', # only outliers
                            marker_color='rgb(107,174,214)',
                            line_color='rgb(107,174,214)',
                            marker_size=10, # increase the size of the plotted points
                            line_width=2 # increase the line width
                        ))
                        # Add annotations to the plot to show the mean and standard deviation
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
                                dash='dash'  # set the dash style to 'dash'
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
                                dash='dash'  # set the dash style to 'dash'
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
                                dash='dash'  # set the dash style to 'dash'
                            )
                        )

                        # Add text labels to the plot to show the mean and standard deviation values
                        fig.add_annotation(
                            x=1.5,
                            y=meanY,
                            text=f"Mean: {meanY:.3f}",
                            showarrow=False,
                            font=dict(
                                color='green',
                                size=12
                            )
                        )
                        fig.add_annotation(
                            x=1.5,
                            y=meanY-stdY,
                            text=f"Mean - Std: {meanY-stdY:.3f}",
                            showarrow=False,
                            font=dict(
                                color='black',
                                size=12
                            )
                        )
                        fig.add_annotation(
                            x=1.5,
                            y=meanY+stdY,
                            text=f"Mean + Std: {meanY+stdY:.3f}",
                            showarrow=False,
                            font=dict(
                                color='black',
                                size=12
                            )
                        )            
                                    
                        fig.update_layout(title_text=f"Box Plot Styling Outliers: {dependent_column_name}")

                        # Display the plot in Streamlit
                        st.plotly_chart(fig)      

            except TypeError:
                    st.error(f'❌ [{dependent_column_name}] is categorical/discrete with at least 2 unique values and [{independent_column_name}] is continuous.')  
            
            except ValueError:
                    st.error(f'❌ [{dependent_column_name}] is categorical/discrete with at least 2 unique values and [{independent_column_name}] is continuous.')  

            except AttributeError:
                    st.error(f'❌ [{dependent_column_name}] is categorical/discrete with at least 2 unique values and [{independent_column_name}] is continuous.')  
