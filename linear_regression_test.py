import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functions import normalize_numpy

from scipy.stats import pearsonr
from scipy.stats import linregress

from sklearn.linear_model import LinearRegression

from streamlit_extras.colored_header import colored_header

def linear_regression(data, file,column):

    data = data.select_dtypes(include=['float'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("\n")
    st.header('🧮 Simple Linear Regression')

    if file is not None:
        
        with st.expander("What is Simple linear regression?",expanded=True):   
            st.write("Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables.")
            st.markdown("- ✅ The variables x and y can be any continuous variables that can take on any value within a range. (e.g. height and weight are continuous variables.)")          
            st.markdown('''
                <style>
                [data-testid="stMarkdownContainer"] ul{
                    padding-left:40px;
                }
                </style>
                ''', unsafe_allow_html=True)

        st.subheader("[👁️‍🗨️] Table Preview:")
        st.dataframe(data, height = 200)
       
        button, slr_row, slr_col = st.columns((0.0001,1.5,4.5), gap="small")
        rows = data.shape[0]
        cols = data.shape[1]
        with slr_row:
            st.markdown(f"<span style='color: violet;'>➕ # of rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
        with slr_col:
            st.markdown(f"<span style='color: violet;'>➕ # of columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)
        with button:
            st.write("")

        colored_header(
        label="",
        description="",
        color_name="violet-70",
        )  

        column_names = data.columns.tolist()
        st.write("\n")

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
   
                    elif (len(set(y)) == 1):
                        st.error(f'❌ SELECTION ERROR #6: All values of {y_column_name} column are identical.')
                    else: 
                        numeric_cols = data._get_numeric_data().columns

                        needs_normalization = []
                        for col in numeric_cols:
                            z_scores = (data[col] - data[col].mean()) / data[col].std()
                            if (z_scores.max() - z_scores.min()) > 3: 
                                needs_normalization.append(col)

                        common_cols = set([x_column_name, y_column_name]).intersection(set(needs_normalization))

                        if common_cols:
                            default_values = list(common_cols)
                        else:
                            default_values = []

                        selected_cols = st.sidebar.multiselect("👉 COLUMN TO BE NORMALIZED (for selected 'y' field above):", needs_normalization, default=default_values)

                        df_final = data.copy()
    
                        if selected_cols:
                            method = "Z-Score"
                            numeric_selected_cols = list(filter(lambda col: col in numeric_cols, selected_cols))
                            categorical_selected_cols = list(filter(lambda col: col not in numeric_cols, selected_cols))
                            df_norm = normalize_numpy(df_final, numeric_selected_cols, categorical_selected_cols, method)
                            not_selected_cols = df_final.columns.difference(selected_cols)
                            df_final = pd.concat([df_norm, df_final[not_selected_cols]], axis=1)

                        X = df_final[x_column_name].values
                        y = df_final[y_column_name].values
                                                                        
                        slope, intercept, r_value, p_value, std_err = linregress(X, y)      
                        #mean_x = np.mean(X)
                        mean_y = np.mean(y)

                        median_y = np.median(y)

                        mode_y = df_final[y_column_name].mode()[0]

                        std_y = np.std(y)
                        
                        r_squared = r_value ** 2

                        #sum_xy = np.sum((X-mean_x)*(y-mean_y))
                        #sum_x2 = np.sum( (X-mean_x)**2)
                        
                        st.write("\n")
                        st.subheader("[✍] Linear Regression Test")
                        st.write("\n")
                        SI1, SI2 = st.columns((1,5), gap="small")
                        with SI1:
                            st.write("")
                        with SI2:
                            with st.expander("The Importance of Slope", expanded=False):
                                st.write("In a linear regression, we draw a straight line between two things to see how they're related. The slope of the line tells us how much one thing changes when the other thing changes by one unit.")
                                st.write("")
                                st.write("For example, if we're looking at how studying affects grades, the slope of the line tells us how much a student's grades change for every additional hour they study. If the slope is positive, it means that as students study more, their grades tend to go up. If the slope is negative, it means that as students study more, their grades tend to go down.")
                                st.write("")
                                st.write("The slope is an important part of linear regression because it helps us understand the relationship between the two things we're looking at. We can use the slope to make predictions, like estimating a student's grade if they study for 3 hours.")

                        slope1a, slope2a = st.columns((1,5), gap="small")
                        with slope1a:
                            st.metric("Slope", f"{slope:.2f}")

                        with slope2a:
                            if slope > 0:
                                st.success(f'* With a positive slope of {slope:.2f}, there is a positive relationship between the {x_column_name} and {y_column_name} variables.')
                                st.write(f'  This means that as the {x_column_name} variable increases, the {y_column_name} variable is also likely to increase.')
                            elif slope < 0:
                                st.warning(f'* With a negative slope of {slope:.2f}, there is a negative relationship between the {x_column_name} and {y_column_name} variables.')
                                st.write(f'  This means that as the {x_column_name} variable increases, the {y_column_name} variable is likely to decrease.')
                            else:
                                st.error(f'* With a slope of {slope:.2f}, there is no relationship between the {x_column_name} and {y_column_name} variables.')
                                st.write(f'  This means that the {x_column_name} variable does not have an impact on the {y_column_name} variable.')

                        st.write("\n")
                        st.write("\n")
                        

                        P1, SL2 = st.columns((1,5), gap="small")
                        with P1:
                            st.write("")
                        with SL2:
                            with st.expander("Understanding the Significance Level and P-value", expanded=False):
                                st.write("In a simple linear regression, you can test the relationship between variables, like the number of hours studied and the grades received for an exam.")
                                st.write("")
                                st.write("The R-value and R-squared value indicate the strength of the linear relationship and how much of the grade variation can be explained by the hours studied.")
                                st.write("")
                                st.write("To assess if the relationship is significant, you calculate the p-value. It measures the likelihood that the observed relationship is due to chance.")
                                st.write("")
                                st.write("By setting a significance level (e.g., 0.05), you determine if the p-value is below that threshold. If so, there is evidence of a significant relationship.")
                                st.write("")
                                st.write("However, if the p-value is above the significance level, the relationship is likely not significant, suggesting the hours studied may not have a notable impact on grades.")
                                st.write("")
                                st.write("By performing a simple linear regression, you can assess significance, make predictions, and understand the relationship between variables.")

                        pvalue1a, pvalue2a = st.columns((1,5), gap="small")
                        with pvalue1a:
                            st.metric("P-value",f"{p_value:.2f}")       
                            st.metric("Significance level:",f"{100-p_value:.2f}")
                        with pvalue2a:    
                            if p_value < 0.05:
                                st.success(f'* With p-value of {p_value:.2f} that is less than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are statistically significant.')
                                st.write("There is a less than 5% chance that the observed difference between the groups happened due to natural differences alone.")
                                st.write("This provides strong evidence that the independent variable has a significant impact on the dependent variable.")
                            else:
                                st.warning(f'* With p-value of {p_value:.2f} that is greater than 0.05, it means that the results for the relationship between {x_column_name} and {y_column_name} are not statistically significant.')
                                st.write("There is a greater than or equal to 5% chance that the observed difference between the groups happened due to natural differences alone.")
                                st.write("This suggests that the independent variable does not have a significant impact on the dependent variable.")   
            
                     
                        
                        # st.write("\n")

                        # SE1, SE2 = st.columns((1,5), gap="small")
                        # with SE1:
                        #     st.write("")
                        # with SE2:
                        #     with st.expander("Understanding the Standard Error",expanded=False):   
                        #         st.write("Imagine you're a student and you're taking a test. You've studied hard and you're feeling confident, but you know that you might still make some mistakes. You finish the test and turn it in, and your teacher grades it and gives you a score. But you're not sure how accurate that score really is, because you know you might have made some mistakes.")
                        #         st.write("")
                        #         st.write("This is where the standard error comes in. It's like a measure of how much you might expect your score to vary if you took the same test over and over again. If the standard error is small, it means that your score is probably pretty accurate and you can be confident in it. But if the standard error is large, it means that your score is less reliable and could be off by quite a bit.")
                        #         st.write("")
                        #         st.write("In a similar way, the standard error in a simple linear regression is a measure of how much you might expect the slope of the regression line to vary if you took different samples from the same population. If the standard error is small, it means that the slope of the regression line is probably a good estimate of the true population slope and you can be confident in it. But if the standard error is large, it means that the slope of the regression line is less reliable and could be off by quite a bit.")

                        # std1a, std2a = st.columns((1,5), gap="small")
                        # with std1a:
                        #     st.metric("Standard Error",f"{std_err:.2f}")  
      
                        # with std2a:
                        #     st.info(f'* The standard error is {std_err:.2f}, it measures the accuracy of the estimate of the slope, a smaller standard error means that the estimate of the slope is more accurate.')
                        
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
  
                            st.write("\n")
                            fig = go.Figure(data=go.Scatter(x=X[:, 0], y=y, mode='markers', name="Data points"))
                            fig.add_scatter(x=X[:, 0], y=y_pred, mode="lines", name="Simple Regression Line")
                            title = f"{x_column_name} vs {y_column_name}"
                            fig.update_layout(title=title, xaxis_title=x_column_name, yaxis_title=y_column_name)
                            fig.update_traces(marker=dict(color='red'), line=dict(color='black'))

                            graph1, graph2 = st.columns((7, 4.7), gap="small")
                            with graph1:
                                st.subheader("[📉] Scatter Plot Graph")
                                st.plotly_chart(fig)
                            with graph2:
                                st.write("#\n" * 5)
                                st.info(f"* This scatter plot shows the relationship between the {x_column_name} and {y_column_name} variables. The data points are plotted in red, and the simple linear regression line is plotted in black. The regression line is a statistical model that describes the linear relationship between the two variables. A linear relationship between two variables refers to a straight-line relationship between them. In other words, if you plot the variables on a graph, the points should fall on or near a straight line.")
                            
                            st.subheader("[💡] Graph Insights")    
          
                            r, p = pearsonr(X[:, 0], y)

                            if r > 0:
                                relationship = "positive"
                            else:
                                relationship = "negative"

                            explanation = f"An increase in the {x_column_name} variable is associated with a {'increase' if r > 0 else 'decrease'} in the {y_column_name} variable, and a decrease in the {x_column_name} variable is associated with a {'decrease' if r > 0 else 'increase'} in the {y_column_name} variable."

                            if r > 0:
                                st.success(f"* There is a {relationship} correlation between the {x_column_name} and {y_column_name} variables (r = {r:.2f}, p = {p:.2f}). {explanation}")
                            else:
                                st.error(f"* There is a {relationship} correlation between the {x_column_name} and {y_column_name} variables (r = {r:.2f}, p = {p:.2f}). {explanation}")







