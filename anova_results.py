import streamlit as st
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header
from scipy.stats import f_oneway
from scipy.stats import f


def anova_result(df_final, independent_column_name, dependent_column_name, file, column):
     
    st.subheader("[✍] ANOVA Test")
    
    ind_col_data = df_final[independent_column_name].values
    dep_col_data = df_final[dependent_column_name].values

    try:
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
            # F1, C1 = st.columns((1,5), gap="small")
            # with F1:
            #     st.write("")
            # with C1:
            #     with st.expander("Understanding the F-Statistic and Critical Value",expanded=False):   
            #         st.write("The F-statistic is a measure of how much the variance between the group means differs from the variance within the groups. If the group means are all equal (as in the null hypothesis), then the F-statistic will be close to 1. However, if the group means are not equal, then the F-statistic will be larger.")
            #         st.write("")
            #         st.write("The F-statistic is compared to a critical value to determine whether to reject or fail to reject the null hypothesis. The critical value is determined based on the degrees of freedom (df) associated with the test. The df are calculated based on the number of groups being compared and the sample size.")
            #         st.write("")
            #         st.write("An analogy for the F-statistic and critical value in an ANOVA test could be a baking competition. Imagine you're judging a competition where three contestants have each made a cake. You want to determine if there is a significant difference in the average taste score between the three cakes.")
            #         st.write("")
            #         st.write("The F-statistic would be like a score that tells you how much the average taste score differs between the three cakes, relative to the variation in taste scores within each cake. If all of the cakes tasted the same, then the F-statistic would be close to 1. But if one cake tastes significantly better than the others, the F-statistic would be higher.")
            #         st.write("")
            #         st.write("The critical value would be like a threshold you set before judging the competition. If the F-statistic exceeds the critical value, you'll conclude that there is a significant difference in taste between the three cakes. However, if the F-statistic falls below the critical value, you'll conclude that there is not a significant difference in taste between the cakes.")
            #         st.write("")
            #         st.write("In summary, the F-statistic and critical value in an ANOVA test help us determine whether there is a significant difference between the means of three or more groups.")
                     
            # value_a, value_b = st.columns((1,5), gap="small")                        
            # with value_a:
            #     st.metric("F-stat:", f"{f_value:.2f}")
            #     st.metric("Critical-value:", f"{c_value:.2f}")

            # with value_b:
            #     if f_value > c_value:
            #         st.success("The calculated F ratio is greater than the critical value.")
            #         st.write("This suggests a significant difference between the groups.")
            #         st.write("The observed differences are unlikely to be solely due to random variations or coincidences.")
            #     else:
            #         st.error("The calculated F ratio is not greater than the critical value.")
            #         st.write("This suggests that there is not enough evidence to conclude a significant difference between the groups.")
            #         st.write("The observed differences could reasonably be attributed to random variations or coincidences.")

            st.write("")

            P1, SL2 = st.columns((1,5), gap="small")
            with P1:
                st.write("")
                
            with SL2:
                with st.expander("Understanding the Significance Level and P-value", expanded=False):
                    st.write("The p-value in ANOVA represents the probability that the differences between the groups are due to chance.")
                    st.write("A small p-value (usually less than 0.05) indicates that the differences between the groups are unlikely to be due to chance, and we can reject the null hypothesis that there is no difference between the groups.")
                    st.write("In other words, if the p-value is small, it suggests that there is a significant difference between at least two of the groups.")
                    st.write("")
                    st.write("The significance level in ANOVA works in a similar way as in other statistical tests.")
                    st.write("We set a significance level, usually at 0.05, which represents the maximum probability of making a Type I error (rejecting the null hypothesis when it's actually true).")
                    st.write("If the p-value is less than the significance level, we reject the null hypothesis and conclude that there is a statistically significant difference between at least two of the groups.")
                    st.write("")
                    st.write("To give an analogy, imagine you are comparing the exam scores of three different classes to see if there is a significant difference between them.")
                    st.write("The null hypothesis would be that there is no significant difference in exam scores between the three classes, and the alternative hypothesis would be that there is a significant difference between at least two of the classes.")
                    st.write("")
                    st.write("You would conduct an ANOVA test and obtain a p-value of 0.02.")
                    st.write("This means that there is a 2% chance of observing the differences in exam scores between the three classes due to chance.")
                    st.write("Since the p-value is less than the significance level of 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference in exam scores between at least two of the classes.")
                    st.write("This information could be useful in identifying areas for improvement or making decisions about which class may need additional resources or attention.")
                    
            value_2a, value_2b = st.columns((1,5), gap="small")   
            with value_2a:
                st.metric("P-value:", f"{p_value:.2f}")
                st.metric("Significance level:", f"{100-p_value:.2f}")

            with value_2b:
                if p_value < 0.05:
                    st.success(f'* With p-value of {p_value:.2f} that is less than 0.05, it means that the results for the relationship between "{independent_column_name}" and "{dependent_column_name}" are statistically significant.')

                    st.write("There is a less than 5% chance that the observed difference between the groups happened due to natural differences alone.")
                    st.write("This provides strong evidence that the independent variable has a significant impact on the dependent variable.")
                else:
                    st.warning(f'* With p-value of {p_value:.2f} that is greater than 0.05, it means that the results for the relationship between {independent_column_name} and {dependent_column_name} are not statistically significant.')
                    st.write("There is a greater than or equal to 5% chance that the observed difference between the groups happened due to natural differences alone.")
                    st.write("This suggests that the independent variable does not have a significant impact on the dependent variable.")   

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
                conclusion = f"There is sufficient evidence to suggest that {independent_column_name} is a factor on {dependent_column_name}."
                st.success(f"* P Value is {p_value:.2f} which is less than or equal to 0.05 ({result}); {conclusion}")
            else:
                result = "Fail to Reject Null Hypothesis"
                conclusion = f"There is not sufficient evidence to suggest that {independent_column_name} is a factor on {dependent_column_name}."
                st.error(f"* P Value is {p_value:.2f} which is greater than to 0.05 ({result}); {conclusion}")

            null_hypothesis = f"The independent variable {independent_column_name} has no effect on the dependent variable {dependent_column_name}."
            alternate_hypothesis = f"The independent variable {independent_column_name} has an effect on the dependent variable {dependent_column_name}."
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
                    boxpoints='all', 
                    marker_color='red',
                    line_color='black',
                    marker_size=3, 
                    line_width=2 
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
                st.write("A box plot, shows how a set of data is distributed visually. To visualize the distribution of numerical data and spot trends, patterns, and outliers, statistical analysis frequently uses this technique.")
                st.write("The box represents range between the first quartile (25th percentile) and the third quartile, or the interquartile range (IQR), is shown by the box in the box plot (75th percentile). The first quartile is the percentage that divides the lowest 25% of the data from the remaining data, and the third quartile, the highest 25% of the data.")
                st.write("The median, which is the number separating the lowest 50% from the highest 50% of the data, is the horizontal line inside the box. It gives the data a measure of central tendency and serves as a reliable guide to the typical value in the data set.")

            st.write("The whiskers extend from the box to represent the range of the data, excluding outliers. Outliers are values that are significantly different from the rest of the data and are represented as individual dots outside the whiskers. They can represent errors in data collection, measurement, or data entry and are important to identify as they can skew the results of a statistical analysis.")
            st.markdown(f"<span style='color: black;'>The x-axis of the box plot represents the independent variable </span><span style='color: blue;'>({independent_column_name})</span><span style='color: black;'> which is the variable that you want to examine the relationship between it and the dependent variable. The y-axis represents the dependent variable<span style='color: blue;'>({dependent_column_name})</span><span style='color: black;'> which is the variable you want to analyze its distribution.</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: black;'>In conclusion, the box plot provides a quick and effective way to visualize the distribution of the dependent variable </span><span style='color: blue;'>({dependent_column_name})</span><span style='color: black;'> across different values of the independent variable</span><span style='color: blue;'>({independent_column_name})</span><span style='color: black;'> . By examining the heights and ranges of the boxes, you can gain insights into the relationship between the dependent variable and the independent variable and identify trends, patterns, and outliers in the data.</span>", unsafe_allow_html=True)

    except (UnboundLocalError, ValueError):
        st.error(f'❌ SELECTION ERROR: {dependent_column_name} column might contain categorical/string variables, please select a quantitative column.')
