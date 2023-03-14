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
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.feature_selection import f_regression

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
                        
                        r_squared = r_value ** 2

                        sum_xy = np.sum((X-mean_x)*(y-mean_y))
                        sum_x2 = np.sum( (X-mean_x)**2)
                        
                        st.write("\n")
                        st.subheader("[✍] Linear Regression Test")
                        st.write("\n")
                        SI1, SI2 = st.columns((1,5), gap="small")
                        with SI1:
                            st.write("")
                        with SI2:
                            with st.expander("The Importance of Slope and Intercept",expanded=False):   
                                st.write("In a linear regression, we draw a straight line between two things to see how they're related. The slope of the line tells us how much one thing changes when the other thing changes by one unit.")
                                st.write("")
                                st.write("For example, if we're looking at how studying affects grades, the slope of the line tells us how much a student's grades change for every additional hour they study. If the slope is positive, it means that as students study more, their grades tend to go up. If the slope is negative, it means that as students study more, their grades tend to go down.")
                                st.write("")
                                st.write("The slope is an important part of the linear regression because it helps us understand the relationship between the two things we're looking at. We can use the slope to make predictions, like if a student studies for 3 hours, we can use the slope to estimate what their grade might be.")
                                st.write("")
                                st.write("In addition to the slope, we also look at the intercept, which is where the line crosses the y-axis. The intercept tells us what the predicted value of the response variable (in our example, the predicted grade) would be if the predictor variable (in our example, the number of hours studied) was zero.")
                                st.write("")
                                st.write("Together, the slope and intercept help us understand the relationship between the two things we're looking at and make predictions based on that relationship.")                        
                            
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
                        st.write("\n")
                        st.write("\n")                        
                        R1, R2 = st.columns((1,5), gap="small")
                        with R1:
                            st.write("")
                        with R2:
                            with st.expander("Exploring the Relationship with R-Value and R-Squared Value",expanded=False):   
                                st.write("Let's say we want to know if there's a relationship between the amount of time a student spends studying and their grades. We can use simple linear regression to see if there's a linear relationship between these two variables.")
                                st.write("")
                                st.write("The R-value tells us how strong the linear relationship is between studying and grades. If we get an R-value of 0.8, it means there's a strong positive linear relationship between studying and grades - as a student spends more time studying, their grades tend to improve. On the other hand, if we get an R-value of -0.6, it means there's a moderate negative linear relationship between studying and grades - as a student spends more time studying, their grades tend to decrease.")
                                st.write("")
                                st.write("The R-squared value tells us how much of the variation in grades can be explained by the amount of time spent studying. For example, if we get an R-squared value of 0.64, it means that 64% of the variation in grades can be explained by the amount of time spent studying. This means that studying is a significant predictor of grades, and that other factors may also be influencing grades.")
                                st.write("")
                                st.write("Overall, the R-value and R-squared value help us understand how strong the relationship is between studying and grades, and how much of the variation in grades can be explained by studying. This information can be useful for predicting future grades based on studying habits, or for identifying areas where students may need extra support or resources.")
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

                        # R-squared
                        rsquare1a, rsquare2a = st.columns((1,5), gap="small")
                        with rsquare1a:
                            #st.metric("R-value (coefficient correlation)",f"{r_value:.2f}")       
                            st.metric("R-squared",f"{r_squared:.2f}")  
                            #st.write(r_value)    
                        with rsquare2a:
                            if r_squared > 0.5:
                                st.success(f'* With an R-squared value of {r_squared:.2f} that is greater than 0.5, the linear regression model explains a significant amount of the variability in the {y_column_name} variable based on the {x_column_name} variable. This indicates a strong fit of the model to the data.')
                            elif r_squared > 0.2:
                                st.warning(f'* With an R-squared value of {r_squared:.2f} that is between 0.2 and 0.5, the linear regression model explains some of the variability in the {y_column_name} variable based on the {x_column_name} variable, but there is still a significant amount of unexplained variability. This indicates a moderate fit of the model to the data.')
                            else:
                                st.error(f'* With an R-squared value of {r_squared:.2f} that is less than 0.2, the linear regression model explains very little of the variability in the {y_column_name} variable based on the {x_column_name} variable, indicating a poor fit of the model to the data.')
                        st.write("\n")
                        st.write("\n")
                        P1, SL2 = st.columns((1,5), gap="small")
                        with P1:
                            st.write("")
                        with SL2:
                            with st.expander("Understanding the Significance Level and P-value",expanded=False):   
                                st.write("Let's say you and your classmates are studying for an exam, and you want to know if the number of hours you study is related to the grades you get. To test this relationship, you decide to perform a simple linear regression.")
                                st.write("")
                                st.write("You start by collecting data on the number of hours each student studied and their corresponding grades. You then plot the data on a scatterplot and draw a straight line through the points to represent the relationship between the two variables.")
                                st.write("")
                                st.write("Once you have this line, you can calculate several values that help you understand the strength and significance of the relationship.")
                                st.write("")
                                st.write("The R-value and R-squared value represent how strong the linear relationship is between the number of hours studied and the grades received. The R-value tells you if the relationship is positive or negative, and the R-squared value tells you how much of the variation in grades can be explained by the number of hours studied.")
                                st.write("")
                                st.write("Now, imagine you want to know if the relationship you see is real or if it's just due to chance. This is where the p-value and significance level come in. The p-value is like a probability that tells you how likely it is that the relationship you see between the two variables is real and not just due to chance.")
                                st.write("")
                                st.write("You set the significance level to 0.05, which means that you're willing to accept a 5% chance that the relationship you see is just by chance. If the p-value is less than 0.05, it means that there is evidence of a significant relationship between the number of hours studied and the grades received.")
                                st.write("")
                                st.write("If the p-value is greater than 0.05, it means that the relationship you see is probably not significant and could just be due to chance. In other words, you cannot conclude that the number of hours studied has a significant effect on the grades received.")
                                st.write("")
                                st.write("Therefore, by using a simple linear regression, you can determine if the relationship between the number of hours studied and the grades received is significant or not, which helps you make predictions and understand the relationship between the two variables.")                                                                                    

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
                        st.write("\n")
                        SE1, SE2 = st.columns((1,5), gap="small")
                        with SE1:
                            st.write("")
                        with SE2:
                            with st.expander("Understanding the Standard Error",expanded=False):   
                                 
                                st.write("Imagine you're a student and you're taking a test. You've studied hard and you're feeling confident, but you know that you might still make some mistakes. You finish the test and turn it in, and your teacher grades it and gives you a score. But you're not sure how accurate that score really is, because you know you might have made some mistakes.")
                                st.write("")
                                st.write("This is where the standard error comes in. It's like a measure of how much you might expect your score to vary if you took the same test over and over again. If the standard error is small, it means that your score is probably pretty accurate and you can be confident in it. But if the standard error is large, it means that your score is less reliable and could be off by quite a bit.")
                                st.write("")
                                st.write("In a similar way, the standard error in a simple linear regression is a measure of how much you might expect the slope of the regression line to vary if you took different samples from the same population. If the standard error is small, it means that the slope of the regression line is probably a good estimate of the true population slope and you can be confident in it. But if the standard error is large, it means that the slope of the regression line is less reliable and could be off by quite a bit.")

                                
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







