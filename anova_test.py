import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.stats import f_oneway

def anova(data, file):
        # Select only numeric columns
        data = data.select_dtypes(include=['object','float','int'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader('📝 Analysis of variance (ANOVA)')

        st.dataframe(data)
        st.write(f"Number of rows :{data.shape[0]:.0f}")
        st.write(f"Number of columns :{data.shape[1]:.0f}")    
        # Create an ag-Grid table using the AgGrid component
        #ag_grid = AgGrid(data, height=300)
        
        column_names = data.columns.tolist()
        # Ask the user to select the column names for the independent and dependent variables
        independent_column_name = st.selectbox('➕ Select the column name for the X (independent/CATEGORICAL/DISCRETE) variable:', column_names)
        dependent_column_name = st.selectbox('➕ Select the column name for the y (dependent/CONTINUOUS) variable:', column_names)

        # Check if the selected columns are the same
        if independent_column_name == dependent_column_name:
            st.error("❌ Both columns are the same. Please select different columns.")
        else:           
            try:
                if ((not pd.api.types.is_string_dtype(data[independent_column_name]) and not pd.api.types.is_integer_dtype(data[independent_column_name])) and data[independent_column_name].nunique() < 2):
                    st.error(f'❌ {independent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
                
                elif (not pd.api.types.is_float_dtype(data[dependent_column_name]) and not np.issubdtype(data[dependent_column_name], np.number)):
                    st.error(f'❌ {dependent_column_name} column must be a continuous variable.') 
                
                elif (pd.api.types.is_float_dtype(data[independent_column_name]) and np.issubdtype(data[independent_column_name], np.number)):
                    st.error(f'❌ {independent_column_name} column must be either categorical/discrete data with atleast 2 unique values.')
                
                elif ((pd.api.types.is_string_dtype(data[dependent_column_name]) or pd.api.types.is_integer_dtype(data[dependent_column_name])) or not data[independent_column_name].nunique() > 1):
                    st.error(f'❌ {dependent_column_name} column must be a continuous variable.') 
                
                else:
                    # Perform the ANOVA test
                    st.subheader("✎ [ANOVA Test]")
                    result = f_oneway(*[group[1][dependent_column_name] for group in data.groupby(independent_column_name)])

                    # Extract the F-value and p-value from the result
                    f_value = result.statistic
                    # Extract the p-value from the result
                    p_value = result.pvalue
                    
                    #st.write(f'F-value (F-ratio):', f_value)
                    #st.write(f'F-value (F-ratio): {f_value:.2f}')
                    st.metric("F-value (F-ratio):",f"{f_value:.2f}")
                    if f_value == float('inf'):
                        st.success(f'* The F-value is not defined because one of the groups has a variance of 0 or the groups have a very large difference in sample size.')
                        st.error(f'* The ANOVA test is not appropriate to use and other statistical tests should be considered.')
                    else:
                        #st.write(f'P-value (significance level):', p_value) 
                        # convert p_value to percentage
                        #st.write(f'P-value (significance level): {p_value:.2e}')
                        st.metric("P-value (significance level):",f"{p_value:.2f}")
                        if p_value < 0.05:
                            st.success(f'* The result is statistically significant, which means that there is a less than 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                            st.success(f'* The F-value is large, which indicates that there is a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                            st.success(f'* This suggests that the {independent_column_name} has a significant effect on the {dependent_column_name}.')
                        else:
                            st.error(f'* The result is not statistically significant, which means that there is a greater than or equal to 0.05 chance that the observed difference between the {dependent_column_name} of the different {independent_column_name} groups occurred by chance alone.')
                            st.error(f'* The F-value is small, which indicates that there is not a significant difference between the {dependent_column_name} of the different {independent_column_name} groups.')
                            st.error(f'* This suggests that the {independent_column_name} has no significant effect on the {dependent_column_name}.')

                        st.subheader("🗠[Graph]")
                        # Select the data for the box plot
                        x = data[independent_column_name]
                        y = data[dependent_column_name]

                        # Create a trace for the box plot
                        traceBox = go.Box(x=x, y=y, name=dependent_column_name)

                        # Create a data object with the trace
                        dataBox = [traceBox]
                        st.write(f'➕ Box Plot: {dependent_column_name} (outcome/dependent/output) ')
                        # Create a box plot using plotly
                        
                        # Calculate the summary statistics for the data
                        summary_statsY = data[dependent_column_name].describe()

                        # Extract the mean, median, and standard deviation from the summary statistics for Y
                        meanY = summary_statsY['mean']
                        medianY = summary_statsY['50%']
                        stdY = summary_statsY['std']
                        
                        # Print the summary statistics for Y
                        #st.write(f'Mean:', meanY) # st.write(f"The mean of the data is {mean:.3f}.")
                        st.write(f'Mean: {meanY:.2f}')
                        #write(f'Median:', medianY) # st.write(f"The median of the data is {median:.3f}.")
                        st.write(f'Median: {medianY:.2f}')
                        #st.write(f'Standard Devidation:', stdY) # st.write(f"The standard deviation of the data is {std:.3f}.")
                        st.write(f'Standard Devidation: {stdY:.2f}')

                        # Analyze the results
                        if meanY > medianY:
                            st.info(f'* The mean is higher than the median, which suggests that the data is skewed to the right.')
                        elif meanY < medianY:
                            st.info(f'* The mean is lower than the median, which suggests that the data is skewed to the left.')
                        else:
                            st.info(f'* The mean is equal to the median, which suggests that the data is symmetrical.')

                        if stdY > 1:
                            st.info(f'* The standard deviation is high, which indicates that the data is dispersed.')
                        else:
                            st.info(f'* The standard deviation is low, which indicates that the data is concentrated.')
                                    
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
                    st.error(f'❌ {dependent_column_name} is categorical/discrete with at least 2 unique values and {independent_column_name} is continuous.')  
            
            except ValueError:
                    st.error(f'❌ {dependent_column_name} is categorical/discrete with at least 2 unique values and {independent_column_name} is continuous.')  

            except AttributeError:
                    st.error(f'❌ {dependent_column_name} is categorical/discrete with at least 2 unique values and {independent_column_name} is continuous.')  

    # The F-value is also known as the F-statistic or F-ratio. It is a measure of the variability between group means relative to the variability within each group.
    # The p-value is the probability of obtaining a result as extreme as the observed result, assuming that the null hypothesis is true. It is used to determine the 
    # statistical significance of the results of a hypothesis test.

    # ANOVA (analysis of variance) is a statistical test that is used to compare the means of two or more groups. It can be used with continuous variables as well 
    # as categorical variables. 

    # If the independent variable is a continuous variable, ANOVA can be used to test whether there is a significant difference in the mean of the dependent variable
    # among the different levels of the independent variable.

    # If the independent variable is a categorical variable with two or more categories, ANOVA can be used to test whether there is a significant difference in the 
    # mean of the dependent variable among the different categories.

    # A dependent variable: This is the variable that is being measured or observed in the study. The dependent variable should be continuous (e.g., height, weight, 
    # income).

    # An independent variable: This is the variable that is being manipulated or controlled in the study. The independent variable should be categorical (e.g., 
    # gender, treatment group).

    # Multiple groups: The independent variable should have at least two groups, and the dependent variable should be measured for each group. For example, if the 
    # independent variable is gender (male vs. female), then the dependent variable (e.g., height) would be measured for both males and females.

    #For example, suppose you are interested in studying the effect of different teaching methods on students' test scores. The independent variable is the teaching 
    #method (with three categories: "lecture-based", "project-based", and "flipped classroom"), and the dependent variable is the test score. In this case, you can
    #use ANOVA to test whether there is a significant difference in the mean test scores among the three teaching methods.
    #In general, ANOVA is a useful tool for comparing the means of different groups and determining whether the differences are statistically significant.
    #However, it is important to note that ANOVA only tests for differences in means, and it does not provide information about the direction or size of the
    # differences.
