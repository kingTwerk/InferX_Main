import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

from scipy.stats import f_oneway
from scipy.stats import pearsonr
from scipy.stats import linregress
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from streamlit_extras.colored_header import colored_header
import os
import datetime

def linear_regression(data, file):

    data = data.select_dtypes(include=['float'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("\n")
    st.header('Single Linear Regression')

    # Check if a file was uploaded
    if file is not None:
 
        st.info("Non-numerical columns are removed.")
        st.dataframe(data)
      
        st.subheader("[👁️‍🗨️] Table Preview:")

        slr_row, slr_col = st.columns((1,5), gap="small")
        rows = data.shape[0]
        cols = data.shape[1]
        with slr_row:
            st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
        with slr_col:
            st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)

        colored_header(
        label="",
        description="",
        color_name="violet-70",
        )  

        column_names = data.columns.tolist()
        st.write("\n")
        x_column_name = st.selectbox('➕ Select the column name for the X (independent/CONTINUOUS) variable:', column_names)
        st.write("\n")
        y_column_name = st.selectbox('➕ Select the column name for the y (dependent/CONTINUOUS) variable:', column_names)
        st.write("\n")
                
        if x_column_name == y_column_name:
            st.error("❌ Both columns are the same. Please select different columns.")
        else:
            try:
                if (not pd.api.types.is_float_dtype(data[x_column_name]) and not np.issubdtype(data[x_column_name], np.number)):
                    st.error(f'❌ {x_column_name} column must be a continuous variable.')
                
                elif (not pd.api.types.is_float_dtype(data[y_column_name]) and not np.issubdtype(data[y_column_name], np.number)):
                    st.error(f'❌ {y_column_name} column must be a continuous variable.')
                
                elif ((pd.api.types.is_string_dtype(data[x_column_name]) and pd.api.types.is_integer_dtype(data[x_column_name]))):
                    st.error(f'❌ {x_column_name} column must be a continuous variable.')                

                elif ((pd.api.types.is_string_dtype(data[y_column_name]) and pd.api.types.is_integer_dtype(data[y_column_name]))):
                    st.error(f'❌ {y_column_name} column must be a continuous variable.')                
                else:
                    X = data[x_column_name].values
                    y = data[y_column_name].values

                    slope, intercept, r_value, p_value, std_err = linregress(X, y)       
                        
                    mean_x = np.mean(X)
                    mean_y = np.mean(y)

                    median_y = np.median(y)

                    mode_y = data[y_column_name].mode()[0]

                    std_y = np.std(y)

                    sum_xy = np.sum((X-mean_x)*(y-mean_y))
                    sum_x2 = np.sum( (X-mean_x)**2)
                    st.write("\n")
                    st.subheader("[✍] Linear Regression Test")
             
                    st.write("\n")
                    slope1a, slope2a = st.columns((1,5), gap="small")
                    with slope1a:
                        st.metric("Slope",f"{slope:.2f}")
                    with slope2a:
                        if slope > 0:
                            st.success(f'* With a slope of {slope:.2f} that is greater than 0, there is a positive relationship between the {x_column_name} and {y_column_name} variables, which means that as the {x_column_name} variable increases, the {y_column_name} variable is also likely to increase.')
                        elif slope < 0:
                            st.warning(f'* With a slope of {slope:.2f} that is less than 0, there is a negative relationship between the {x_column_name} and {y_column_name} variables, which means that as the {x_column_name} variable increases, the {y_column_name} variable is likely to decrease.')
                        else:
                            st.error(f'* There is no relationship between the {x_column_name} and {y_column_name} variables, which means that the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                    intercept1a, intercept2a = st.columns((1,5), gap="small")
                    with intercept1a:
                        st.metric("Intercept",f"{intercept:.2f}")
                    with intercept2a:
                        if intercept > 0:
                            st.success(f'* With an intercept of {intercept:.2f} that is greater than 0, it means that when the {x_column_name} variable is zero, the predicted value of the {y_column_name} variable is {intercept:.2f}')
                        else:
                            st.warning(f'* With an intercept of {intercept:.2f} that is less than 0, it means that when the {x_column_name} variable is zero, the predicted value of the {y_column_name} variable is {intercept:.2f}')

                    rvalue1a, rvalue2a = st.columns((1,5), gap="small")
                    with rvalue1a:
                        st.metric("R-value (coefficient correlation)",f"{r_value:.2f}")           
                    with rvalue2a:
                        if r_value > 0:
                            st.success(f'* With R-value of {r_value:.2f} that is greater than 0, it means that there is a positive correlation between {x_column_name} and {y_column_name} variables, as the {x_column_name} variable increases, the {y_column_name} variable also increases.')
                        elif r_value < 0:
                            st.warning(f'* With R-value of {r_value:.2f} that is less than 0, it means that there is a negative correlation between {x_column_name} and {y_column_name} variables, as the {x_column_name} variable increases, the {y_column_name} variable decreases.')
                        else:
                            st.error(f'* With R-value of {r_value:.2f} that is equal to 0, it means that there is no correlation between {x_column_name} and {y_column_name} variables, the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                    pvalue1a, pvalue2a = st.columns((1,5), gap="small")
                    with pvalue1a:
                        st.metric("P-value (Significance level)",f"{p_value:.2f}")        
                        st.write(p_value)
                    with pvalue2a:    
                        if p_value < 0.05:
                            st.success(f'* With p-value of {p_value:.2f} that is less than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are statistically significant, it means that the probability of the results being by chance is less than 5%. So we can say that the relationship between the variables is not random.')
                        else:
                            st.warning(f'* With p-value of {p_value:.2f} that is greater than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are not statistically significant, it means that the probability of the results being by chance is greater than 5%. So we can say that the relationship between the variables is random.')
                 
                    std1a, std2a = st.columns((1,5), gap="small")
                    with std1a:
                        st.metric("Standard Error",f"{std_err:.2f}")  
                    with std2a:
                        st.info(f'* The standard error is {std_err:.2f}, it measures the accuracy of the estimate of the slope, a smaller standard error means that the estimate of the slope is more accurate.')
                    
                    colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )  
 
                    predictions = intercept + slope * X

                    if pd.isna(X).any():
                        st.warning(f'❌ The column "{x_column_name}" contains blank values, represented either by N/A, -, *, or alike. LinearRegression does not accept missing values.')
                    elif pd.isna(y).any():
                        st.warning(f'❌ The column "{y_column_name}" contains blank values, represented either by N/A, -, *, or alike. LinearRegression does not accept missing values.')
                    else:
                        model = LinearRegression()
                        X = data[x_column_name].values.reshape(-1, 1)
                        y = data[y_column_name].values
                        model.fit(X, y)

                        y_pred = model.predict(X)

                        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=y, mode='markers', name='Data Points'))

                        fig.add_scatter(x=X[:, 0], y=y_pred, mode="lines", name="Linear Regression Line")
                        
                        title = f"{x_column_name} vs {y_column_name}"
                        fig.update_layout(title=title)
                        
                        fig.update_traces(marker=dict(color='skyblue'))

                        fig.update_traces(line=dict(color='lightgreen'))

                        st.write("\n")
                        st.subheader("[📝] Descriptive Statistics for Y")
                        st.write("\n")
                      
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
                        
                        if p_value <= 0.05:
                            result = "Reject Null Hypothesis"
                            conclusion = "{} is a factor on {}.".format(x_column_name, y_column_name)
                            st.info("* P Value is {:.2f} which is less than or equal to 0.05 ({}); {}".format(p_value, result, conclusion))
                        else:
                            result = "Fail to Reject Null Hypothesis"
                            conclusion = "{} is not a factor on {}.".format(x_column_name, y_column_name)
                            st.warning("* P Value is {:.2f} which is greater than to 0.05 ({}); {}".format(p_value, result, conclusion))

                        colored_header(
                        label="",
                        description="",
                        color_name="violet-70",
                        )                  
                        # Display the plot in Streamlit
                        st.write("\n")
                        
                        graph1, graph2 = st.columns((7,4.7), gap="small")
                         
                        with graph1:
                            st.subheader("[📉] Scatter Plot Graph")
                            st.plotly_chart(fig)
                        with graph2:     
                            st.subheader("[💡] Graph Insights") 

                            st.info(f"* This scatter plot shows the relationship between the {x_column_name} and {y_column_name} variables. The data points are plotted in sky blue, and the linear regression line is plotted in dark blue. The regression line is a statistical model that describes the linear relationship between the two variables. A linear relationship between two variables refers to a straight-line relationship between them. In other words, if you plot the variables on a graph, the points should fall on or near a straight line.")
                            st.write("\n")
                            
                            r, p = pearsonr(X[:, 0], y)
                            if r > 0:
                                relationship = "positive"
                                explanation = f"An increase in the {x_column_name} variable is associated with an increase in the {y_column_name} variable, and a decrease in the {x_column_name} variable is associated with a decrease in the {y_column_name} variable."
                                st.success(f"* There is a {relationship} correlation between the {x_column_name} and {y_column_name} variables (r = {r:.2f}, p = {p:.2e}). {explanation}")    
                            else:
                                relationship = "negative"
                                explanation = f"An increase in the {x_column_name} variable is associated with a decrease in the {y_column_name} variable, and a decrease in the {x_column_name} variable is associated with an increase in the {y_column_name} variable."
                                st.error(f"* There is a {relationship} correlation between the {x_column_name} and {y_column_name} variables (r = {r:.2f}, p = {p:.2e}). {explanation}")  
                
            except TypeError:
                st.error(f'❌ Both [{x_column_name}] and [{y_column_name}] columns need to be continuous values.')  
            except ValueError:
                st.error(f'❌ Both [{x_column_name}] and [{y_column_name}] columns need to be continuous values.')     
            except AttributeError:
                st.error(f'❌ Both [{x_column_name}] and [{y_column_name}] columns need to be continuous values.')  
