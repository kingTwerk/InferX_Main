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

def linear_regression(data, file):
    # Select only numeric columns
    data = data.select_dtypes(include=['float'])
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('📝 Single Linear Regression')

    # Check if a file was uploaded
    if file is not None:
        #  file_name = file.name
            
        # Display the contents in a scrollable table
        st.info("Non-numerical columns are removed.")
        st.dataframe(data)
        st.write(f"Number of rows :{data.shape[0]:.0f}")
        st.write(f"Number of columns :{data.shape[1]:.0f}")
        #ag_grid = AgGrid(data, height=300)
        
        column_names = data.columns.tolist()
        # Ask the user to select the column names for the X and y variables
        x_column_name = st.selectbox('➕ Select the column name for the X (independent/CONTINUOUS) variable:', column_names)
        y_column_name = st.selectbox('➕ Select the column name for the y (dependent/CONTINUOUS) variable:', column_names)

        # Check if the selected columns are the same
                
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
                    # Perform the Single Linear Regression test    
                    # Extract the values of the X and y variables
                    X = data[x_column_name].values
                    y = data[y_column_name].values

                    # Assume x and y are your data
                    slope, intercept, r_value, p_value, std_err = linregress(X, y)       
                        
                    # Calculate the mean of X and y
                    mean_x = np.mean(X)
                    mean_y = np.mean(y)

                    # calculate sum of x and y
                    sum_xy = np.sum((X-mean_x)*(y-mean_y))
                    sum_x2 = np.sum( (X-mean_x)**2)
                    
                    st.subheader("✎ [Linear Regression Test]")
                    #input_val_x = st.number_input('➕ Enter the input value (x):')
                    input_val_x = 1
                    y_pred = intercept + slope * input_val_x

                    # Display the results
                    # Slope
                    st.metric("Slope",f"{slope:.2f}")
                    if slope > 0:
                        st.success(f'* With a slope of {slope:.2f} that is greater than 0, there is a positive relationship between the {x_column_name} and {y_column_name} variables, which means that as the {x_column_name} variable increases, the {y_column_name} variable is also likely to increase.')
                    elif slope < 0:
                        st.warning(f'* With a slope of {slope:.2f} that is less than 0, there is a negative relationship between the {x_column_name} and {y_column_name} variables, which means that as the {x_column_name} variable increases, the {y_column_name} variable is likely to decrease.')
                    else:
                        st.error(f'* There is no relationship between the {x_column_name} and {y_column_name} variables, which means that the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                    # Intercept
                    st.metric("Intercept",f"{intercept:.2f}")
                    if intercept > 0:
                        st.success(f'* With an intercept of {intercept:.2f} that is greater than 0, it means that when the {x_column_name} variable is zero, the predicted value of the {y_column_name} variable is {intercept:.2f}')
                    else:
                        st.warning(f'* With an intercept of {intercept:.2f} that is less than 0, it means that when the {x_column_name} variable is zero, the predicted value of the {y_column_name} variable is {intercept:.2f}')

                    # R-value
                    st.metric("R-value (Pearson coefficient of correlation)",f"{r_value:.2f}")           
                    if r_value > 0:
                        st.success(f'* With R-value of {r_value:.2f} that is greater than 0, it means that there is a positive correlation between {x_column_name} and {y_column_name} variables, as the {x_column_name} variable increases, the {y_column_name} variable also increases.')
                    elif r_value < 0:
                        st.warning(f'* With R-value of {r_value:.2f} that is less than 0, it means that there is a negative correlation between {x_column_name} and {y_column_name} variables, as the {x_column_name} variable increases, the {y_column_name} variable decreases.')
                    else:
                        st.error(f'* With R-value of {r_value:.2f} that is equal to 0, it means that there is no correlation between {x_column_name} and {y_column_name} variables, the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                    # P-value
                    st.metric("P-value (Significance level)",f"{p_value:.2e}")              
                    if p_value < 0.05:
                        st.success(f'* With p-value of {p_value:.2e} that is less than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are statistically significant, it means that the probability of the results being by chance is less than 5%. So we can say that the relationship between the variables is not random.')
                    else:
                        st.warning(f'* With p-value of {p_value:.2e} that is greater than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are not statistically significant, it means that the probability of the results being by chance is greater than 5%. So we can say that the relationship between the variables is random.')
                        
                    # Standard error
                    st.metric("Standard Error",f"{std_err:.2f}")  
                    st.info(f'* The standard error is {std_err:.2f}, it measures the accuracy of the estimate of the slope, a smaller standard error means that the estimate of the slope is more accurate.')

                    # Intercept standard error
                    #st.metric("Standard Error of the Estimated Intercept", f"{intercept_stderr:.2f}")
                    #if intercept_stderr < 0.1:
                    #    st.success(f'* The standard error of the estimated intercept is {intercept_stderr:.2f}, which is less than 0.1. This means that the estimate of the intercept is very accurate.')
                    #else:
                    #    st.warning(f'* The standard error of the estimated intercept is {intercept_stderr:.2f}, which is greater than 0.1. This means that the estimate of the intercept is less accurate.')
                                    
                    # Predict the output for each input value using the regression line
                    predictions = intercept + slope * X

                    st.subheader("🗠[Graph]")
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
                        fig = go.Figure(data=go.Scatter(x=X[:, 0], y=y, mode='markers', name='Data Points'))

                        # Add the linear regression line to the plot
                        fig.add_scatter(x=X[:, 0], y=y_pred, mode="lines", name="Linear Regression Line")
                        
                        # Set the title of the plot using the column names of the X and y variables
                        title = f"{x_column_name} vs {y_column_name}"
                        fig.update_layout(title=title)
                        
                        # Set the color of the data points to sky blue
                        fig.update_traces(marker=dict(color='skyblue'))

                        # Set the color of the linear regression line to dark blue
                        fig.update_traces(line=dict(color='lightgreen'))
                        
                        # Display the plot in Streamlit
                        st.plotly_chart(fig)
                                            
                        st.info(f"* This scatter plot shows the relationship between the {x_column_name} and {y_column_name} variables. The data points are plotted in sky blue, and the linear regression line is plotted in dark blue. The regression line is a statistical model that describes the linear relationship between the two variables. A linear relationship between two variables refers to a straight-line relationship between them. In other words, if you plot the variables on a graph, the points should fall on or near a straight line.")
                        st.subheader("💡[Graph Insights]")     

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
            
            except TypeError:
                st.error(f'❌ Both [{x_column_name}] and [{y_column_name}] columns need to be continuous values.')  
            except ValueError:
                st.error(f'❌ Both [{x_column_name}] and [{y_column_name}] columns need to be continuous values.')     
            except AttributeError:
                st.error(f'❌ Both [{x_column_name}] and [{y_column_name}] columns need to be continuous values.')  
 








        
    # Linear regression is a statistical method used to model the linear relationship between a dependent variable and one or more 
    # independent variables. In order to perform linear regression, the following variables are required: 

    # A dependent variable: This is the variable that is being predicted or explained by the model. The dependent variable should be 
    # continuous (e.g., height, weight, income). One or more independent variables: These are the variables that are used to predict 
    # the value of the dependent variable. 

    # Independent variables can be continuous or categorical. Linear regression assumes that  there is a linear relationship between
    # the dependent variable and the independent variables. This means that the change in the dependent variable can be predicted from
    # a change in the independent variables using a straight line. The linear regression model estimates the slope and intercept of 
    # this line based on the data, and can be used to make predictions about the dependent variable given certain values of the independent variables.
