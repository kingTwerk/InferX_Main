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

def linear_regression(data, file,column):

    data = data.select_dtypes(include=['float','int'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("\n")
    st.header('Simple Linear Regression')

    if file is not None:
        
        with st.expander("Simple linear regression?",expanded=True):   
            st.write("Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables.")
            st.write("")
            st.write("The variables x and y do not need to be non-numeric variables. They can be any continuous variables that can take on any value within a range.")
            st.markdown("- e.g., height and weight are continuous variables.")
            st.write("")
            st.write("Categorical and nominal variables are not suitable for simple linear regression because they cannot be measured on a numerical scale.")
            st.markdown("- e.g., gender and eye color are categorical variables.")             
            st.markdown('''
                <style>
                [data-testid="stMarkdownContainer"] ul{
                    padding-left:40px;
                }
                </style>
                ''', unsafe_allow_html=True)
        st.info("Non-numerical columns are removed.")
        st.subheader("[👁️‍🗨️] Table Preview:")
        st.dataframe(data)

        button, slr_row, slr_col = st.columns((5,1,1), gap="small")
        rows = data.shape[0]
        cols = data.shape[1]
        with slr_row:
            st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
        with slr_col:
            st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)
        with button:
            st.write("")
            #if st.button("Download CSV"):
            #    data = data.select_dtypes(include=['float'])
            #    now = datetime.datetime.now()
            #    date_string = now.strftime("%Y-%m-%d")
            #    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            #    save_path = os.path.join(desktop, f'linear_filtered_data_csv_{date_string}.csv')
            #    data.to_csv(save_path, index=False)
            #    st.success(f'File saved successfully to {save_path}!')

        colored_header(
        label="",
        description="",
        color_name="violet-70",
        )  

        column_names = data.columns.tolist()
        st.write("\n")

        #x_column_name = st.selectbox('➕ Select the column name for the X (independent/CONTINUOUS) variable:', column_names)
        float_column_names = [col for col in column_names if data[col].dtype == 'float64']
        x_column_name = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", float_column_names)
        y_column_name = column
        if x_column_name == y_column_name:
            st.error("ATTENTION: ❌ Both columns are the same. Please select different columns.")
        else:
                if (not pd.api.types.is_float_dtype(data[x_column_name]) and not np.issubdtype(data[x_column_name], np.number)):
                    st.error(f'❌ SELECTION ERROR #1: {x_column_name} column must be a continuous variable.')
                
                elif (not pd.api.types.is_float_dtype(data[y_column_name]) and not np.issubdtype(data[y_column_name], np.number)):
                    st.error(f'❌ SELECTION ERROR #2: {y_column_name} column must be a continuous variable.')
                
                elif ((pd.api.types.is_string_dtype(data[x_column_name]) and pd.api.types.is_integer_dtype(data[x_column_name]))):
                    st.error(f'❌ SELECTION ERROR #3: {x_column_name} column must be a continuous variable.')                

                elif ((pd.api.types.is_string_dtype(data[y_column_name]) and pd.api.types.is_integer_dtype(data[y_column_name]))):
                    st.error(f'❌ SELECTION ERROR #4: {y_column_name} column must be a continuous variable.')             
                else:
                    X = data[x_column_name].values
                    y = data[y_column_name].values
                    
                    if (len(set(X)) == 1):
                        st.error(f'❌ SELECTION ERROR #5: All values of {x_column_name} column are identical.')
                        #st.error("❌ SELECTION ERR. #5: All values of x and y are identical, cannot calculate linear regression.") 
                    elif (len(set(y)) == 1):
                        st.error(f'❌ SELECTION ERROR #6: All values of {y_column_name} column are identical.')
                        #st.error("❌ SELECTION ERR. #6: All values of x and y are identical, cannot calculate linear regression.")    
                    else:                           
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
                            #st.write(slope)
                        with slope2a:
                            if slope > 0:
                                st.success(f'* With a slope of {slope:.2f} that is greater than 0, there is a positive relationship between the {x_column_name} and {y_column_name} variables, which means that as the {x_column_name} variable increases, the {y_column_name} variable is also likely to increase.')
                            elif slope < 0:
                                st.warning(f'* With a slope of {slope:.2f} that is less than 0, there is a negative relationship between the {x_column_name} and {y_column_name} variables, which means that as the {x_column_name} variable increases, the {y_column_name} variable is likely to decrease.')
                            else:
                                st.error(f'* There is no relationship between the {x_column_name} and {y_column_name} variables, which means that the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                        # Intercept
                        intercept1a, intercept2a = st.columns((1,5), gap="small")
                        with intercept1a:
                            st.metric("Intercept",f"{intercept:.2f}")
                            #st.write(intercept)
                        with intercept2a:
                            if intercept > 0:
                                st.success(f'* With an intercept of {intercept:.2f} that is greater than 0, it means that when the {x_column_name} variable is zero, the predicted value of the {y_column_name} variable is {intercept:.2f}')
                            else:
                                st.warning(f'* With an intercept of {intercept:.2f} that is less than 0, it means that when the {x_column_name} variable is zero, the predicted value of the {y_column_name} variable is {intercept:.2f}')

                        # R-value
                        rvalue1a, rvalue2a = st.columns((1,5), gap="small")
                        with rvalue1a:
                            #st.metric("R-value (coefficient correlation)",f"{r_value:.2f}")       
                            st.metric("R-value",f"{r_value:.2f}")  
                            #st.write(r_value)    
                        with rvalue2a:
                            if r_value > 0:
                                st.success(f'* With R-value of {r_value:.2f} that is greater than 0, it means that there is a positive correlation between {x_column_name} and {y_column_name} variables, as the {x_column_name} variable increases, the {y_column_name} variable also increases.')
                            elif r_value < 0:
                                st.warning(f'* With R-value of {r_value:.2f} that is less than 0, it means that there is a negative correlation between {x_column_name} and {y_column_name} variables, as the {x_column_name} variable increases, the {y_column_name} variable decreases.')
                            else:
                                st.error(f'* With R-value of {r_value:.2f} that is equal to 0, it means that there is no correlation between {x_column_name} and {y_column_name} variables, the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                        # P-value
                        pvalue1a, pvalue2a = st.columns((1,5), gap="small")
                        with pvalue1a:
                            st.metric("P-value",f"{p_value:.2f}")        
                            #st.write(p_value)
                            st.metric("Significance level:",f"{100-p_value:.2f}")
                        with pvalue2a:    
                            if p_value < 0.05:
                                st.success(f'* With p-value of {p_value:.2f} that is less than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are statistically significant, it means that the probability of the results being by chance is less than 5%. So we can say that the relationship between the variables is not random.')
                            else:
                                st.warning(f'* With p-value of {p_value:.2f} that is greater than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are not statistically significant, it means that the probability of the results being by chance is greater than 5%. So we can say that the relationship between the variables is random.')
                                
                        # Standard error
                        std1a, std2a = st.columns((1,5), gap="small")
                        with std1a:
                            st.metric("Standard Error",f"{std_err:.2f}")  
                            #st.write(std_err)
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
                            # Fit a linear regression model using scikit-learn
                            model = LinearRegression()
                            X = data[x_column_name].values.reshape(-1, 1)
                            y = data[y_column_name].values
                            model.fit(X, y)

                            # Get the predicted y values
                            y_pred = model.predict(X)

                            # Create a scatter plot using plotly
                            #fig = px.scatter(data, x=x_column_name, y=y_column_name)
                            fig = go.Figure(data=go.Scatter(x=X[:, 0], y=y, mode='markers', name="Data points"))

                            # Add the linear regression line to the plot
                            fig.add_scatter(x=X[:, 0], y=y_pred,mode="lines", name="Simple Regression Line")
                            
                            # Set the title of the plot using the column names of the X and y variables
                            title = f"{x_column_name} vs {y_column_name}"
                            fig.update_layout(title=title)

                            fig.update_layout(
                                    xaxis_title=x_column_name,
                                    yaxis_title=y_column_name
                                )
                            
                            # Set the color of the data points to sky blue
                            fig.update_traces(marker=dict(color='red'))

                            # Set the color of the linear regression line to dark blue
                            fig.update_traces(line=dict(color='black'))

                            st.write("\n")
                            st.subheader("[📝] Descriptive Statistics for Y")
                            st.write("\n")
                            #st.write(f'Mean: {mean_y:.2f}')
                            mean1a, mean2a = st.columns((1,5), gap="small")
                            with mean1a:
                                st.metric("Mean:",f"{mean_y:.2f}")
                                #st.write(mean_y)
                            with mean2a:
                                st.info("* The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")
                            median1a, median2a = st.columns((1,5), gap="small")
                            with median1a:
                                st.metric("Median:",f"{median_y:.2f}")
                                #st.write(median_y)
                            with median2a:
                                st.info("* The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
                            mode1a, mode2a = st.columns((1,5), gap="small")
                            with mode1a:
                                st.metric("Mode:",f"{mode_y:.2f}")
                                #st.write(mode_y)
                            with mode2a:    
                                st.info("* The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")
                            std1a, std2a = st.columns((1,5), gap="small")
                            with std1a:
                                st.metric("Standard Deviation:",f"{std_y:.2f}")
                                #st.write(std_y)
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
                            if p_value <= 0.05:
                                result = "Reject Null Hypothesis"
                                conclusion = "There is sufficient evidence to suggest that {} is a factor on {}.".format(x_column_name, y_column_name)
                                st.info("* P Value is {:.2f} which is less than or equal to 0.05 ({}); {}".format(p_value, result, conclusion))
                            else:
                                result = "Fail to Reject Null Hypothesis"
                                conclusion = "There is not sufficient evidence to suggest that {} is a factor on {}.".format(x_column_name, y_column_name)
                                st.warning("* P Value is {:.2f} which is greater than to 0.05 ({}); {}".format(p_value, result, conclusion))

                            # Add null and alternate hypothesis statements
                            null_hypothesis = "The independent variable {} has no effect on the dependent variable {}.".format(x_column_name, y_column_name)
                            alternate_hypothesis = "The independent variable {} has an effect on the dependent variable {}.".format(x_column_name, y_column_name)
                            st.write("\n\n")
                            st.markdown(f"<span style='color: blue;'>Null Hypothesis (H0): </span> <span style='color: black;'>{null_hypothesis}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span style='color: blue;'>Alternate Hypothesis (H1): </span> <span style='color: black;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

                            st.write("\n\n")
                            st.markdown(f"<span style='color: black;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: blue;'>({x_column_name})</span> <span style='color: black;'> has an effect on the dependent variable </span> <span style='color: blue;'>({y_column_name})</span>. <span style='color: black;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: blue;'>({x_column_name})</span> <span style='color: black;'>not have an effect on the dependent variable </span> <span style='color: blue;'>({y_column_name})</span><span style='color: black;'>.</span>", unsafe_allow_html=True)
                    
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
                                st.write("#")
                                st.write("#")
                                st.write("#")
                                st.write("#")
                                st.write("")
                                st.info(f"* This scatter plot shows the relationship between the {x_column_name} and {y_column_name} variables. The data points are plotted in red, and the simple linear regression line is plotted in black. The regression line is a statistical model that describes the linear relationship between the two variables. A linear relationship between two variables refers to a straight-line relationship between them. In other words, if you plot the variables on a graph, the points should fall on or near a straight line.")
                                st.write("\n")

                            st.subheader("[💡] Graph Insights")    
                            # Determine the kind of relationship between the two variables
                            r, p = pearsonr(X[:, 0], y)
                            if r > 0:
                                relationship = "positive"
                                explanation = f"An increase in the {x_column_name} variable is associated with an increase in the {y_column_name} variable, and a decrease in the {x_column_name} variable is associated with a decrease in the {y_column_name} variable."
                                st.success(f"* There is a {relationship} correlation between the {x_column_name} and {y_column_name} variables (r = {r:.2f}, p = {p:.2e}). {explanation}")    
                            else:
                                relationship = "negative"
                                explanation = f"An increase in the {x_column_name} variable is associated with a decrease in the {y_column_name} variable, and a decrease in the {x_column_name} variable is associated with an increase in the {y_column_name} variable."
                                st.error(f"* There is a {relationship} correlation between the {x_column_name} and {y_column_name} variables (r = {r:.2f}, p = {p:.2e}). {explanation}")  







