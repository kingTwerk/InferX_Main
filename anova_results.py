import streamlit as st
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header
from scipy.stats import f_oneway
from scipy.stats import f
from scipy.stats import ttest_ind
import statsmodels.api as sm
import pingouin as pg
import scipy.stats as stats
from scipy.stats import pearsonr

from pingouin import anova
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def anova_result(df_final, independent_column_name, dependent_column_name, file, column):
     
    st.subheader("One-Way ANOVA")
    
    #ind_col_data = df_final[independent_column_name].values
    #dep_col_data = df_final[dependent_column_name].values

    try:
        # OLD
        #result = f_oneway(ind_col_data, dep_col_data)
        #f_value = result.statistic
        #p_value = result.pvalue

        #alpha = 0.05
        #k = len(df_final[independent_column_name].unique())
        #dfn = k-1

        #n = len(df_final)
        #dfd = n-k
        #c_value = f.ppf(1-alpha, dfn, dfd)   

        # NEW
        groups = df_final.groupby(independent_column_name)[dependent_column_name].apply(list)
        groupsf_value, groupsp_value = stats.f_oneway(*groups)
        # st.write(f"Dependent Variable: {dependent_column_name}")
        # st.write(f"Independent Variable: {independent_column_name}")
        # st.write("p-value:", round(groupsp_value, 3))
        # st.write("F-value:", round(groupsf_value, 3))

        if groupsf_value == float('inf'):
            st.success(f'* The F-value is not defined because one of the groups has a variance of 0 or the groups have a very large difference in sample size.')
            st.error(f'* The ANOVA test is not appropriate to use and other statistical tests should be considered.')
        else:
            st.write("\n")

            with st.expander("Understanding the Significance Level and P-value", expanded=False):
                st.write("In ANOVA (Analysis of Variance), the significance level and p-value are important measures used to determine if there are statistically significant differences between groups.")

                st.write("The p-value represents the probability that the observed differences between the groups are due to chance alone. A small p-value, typically less than 0.05, indicates that the differences between the groups are unlikely to be due to chance alone. When the p-value is small, it suggests that there is a statistically significant difference between at least two of the groups being compared.")

                st.write("The significance level, often set at 0.05, is the threshold we use to make a decision about the statistical significance of the results. It represents the maximum acceptable probability of making a Type I error, which is rejecting the null hypothesis when it's actually true. If the obtained p-value is less than the significance level, typically 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference between at least two of the groups.")

                st.write("To better understand this concept, let's consider an analogy with comparing exam scores of three different classes. The null hypothesis would be that there is no significant difference in exam scores between the three classes, and the alternative hypothesis would be that there is a significant difference between at least two of the classes. By conducting an ANOVA test, we obtain a p-value of 0.02. This means that there is a 2% chance of observing the observed differences in exam scores between the three classes due to chance alone. Since the p-value (0.02) is less than the significance level (0.05), we reject the null hypothesis and conclude that there is a statistically significant difference in exam scores between at least two of the classes.")

                st.write("Understanding the significance level and p-value helps us make informed decisions and draw meaningful conclusions from ANOVA results. It can provide insights into group differences and guide further analysis or decision-making processes, such as identifying areas for improvement or allocating resources to specific groups.")
            
            st.write("\n")

            value_2a, value_2b = st.columns((2,5), gap="small")   
            with value_2a:
                st.metric("P-value:", f"{groupsp_value:.3f}")
            with value_2b:
                st.metric("Significance level:", f"{100-groupsp_value:.3f}")

            if groupsp_value < 0.05:
                st.success(f' With p-value of {groupsp_value:.3f} that is less than 0.05, it means that the results for the relationship between "{independent_column_name}" and "{dependent_column_name}" are statistically significant.')
                st.write("There is a less than 5% chance that the observed difference between the groups happened due to natural differences alone. This provides strong evidence that the independent variable has a significant impact on the dependent variable.")
            else:
                st.error(f' With p-value of {groupsp_value:.3f} that is greater than 0.05, it means that the results for the relationship between "{independent_column_name}" and "{dependent_column_name}" are not statistically significant.')
                st.write("There is a greater than or equal to 5% chance that the observed difference between the groups happened due to natural differences alone.This suggests that the independent variable does not have a significant impact on the dependent variable.")   

            st.write("\n")
            st.subheader("Hypothesis Testing")
            if groupsp_value <= 0.05:
                result = "Reject Null Hypothesis"
                conclusion = f"There is sufficient evidence to suggest that {independent_column_name} is a factor on {dependent_column_name}."
                st.success(f" P Value is {groupsp_value:.3f} which is less than or equal to 0.05 ({result}); {conclusion}")
            else:
                result = "Fail to Reject Null Hypothesis"
                conclusion = f"There is not sufficient evidence to suggest that {independent_column_name} is a factor on {dependent_column_name}."
                st.error(f" P Value is {groupsp_value:.3f} which is greater than to 0.05 ({result}); {conclusion}")

            null_hypothesis = f"The independent variable {independent_column_name} has no effect on the dependent variable {dependent_column_name}."
            alternate_hypothesis = f"The independent variable {independent_column_name} has an effect on the dependent variable {dependent_column_name}."
            st.write("\n\n")
            st.markdown(f"<span style='color: #803df5; font-weight: bold;'>Null Hypothesis (H0): </span> <span style='color: #344a80;'>{null_hypothesis}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #803df5; font-weight: bold;'>Alternate Hypothesis (H1): </span> <span style='color: #344a80;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

            st.write("\n\n")
            st.markdown(f"<span style='color: #344a80;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: #803df5;'>({independent_column_name})</span> <span style='color: #344a80;'> has an effect on the dependent variable </span> <span style='color: #803df5;'>({dependent_column_name})</span>. <span style='color: #344a80;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: #803df5;'>({independent_column_name})</span> <span style='color: #344a80;'>not have an effect on the dependent variable </span> <span style='color: #803df5;'>({dependent_column_name})</span><span style='color: #344a80;'>.</span>", unsafe_allow_html=True)
            x = df_final[independent_column_name]
            y = df_final[dependent_column_name]

            traceBox = go.Box(x=x, y=y, name=dependent_column_name)

            dataBox = [traceBox]

            summary_statsY = df_final[dependent_column_name].describe()

            meanY = summary_statsY['mean']
            medianY = summary_statsY['50%']
            stdY = summary_statsY['std']
            modeY = df_final[dependent_column_name].mode()

            colored_header(
            label="",
            description="",
            color_name="violet-70",
            )
            st.subheader(f"Descriptive Statistics for 'Y' ({dependent_column_name}) ")
            st.write("\n")

            mean1a, median1a, mode1a, std1a = st.columns((2.5,2.5,2.5,2.5), gap="small")
            with mean1a:
                st.metric("Mean:",f"{meanY:.3f}")
            with median1a:
                st.metric("Median:",f"{medianY:.3f}")
            if modeY.shape[0] == 0:
                st.warning("This data set doesn't have a mode.")
            else:
                for i in range(modeY.shape[0]):
                    with mode1a:
                        st.metric("Modes:",f"{modeY[i]:.3f}")  
            with std1a:    
                st.metric("Standard Deviation:",f"{stdY:.3f}")     

            Smean1a, Smedian1a, Smode1a, Sstd1a = st.columns((2.5,2.5,2.5,2.5), gap="small")
            with Smean1a:
                st.info(" The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")
            with Smedian1a:
                st.info(" The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
            if modeY.shape[0] == 0:
                st.warning("This data set doesn't have a mode.")
            else:
                for i in range(modeY.shape[0]):
                    with Smode1a:
                        st.info(" The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")     
            with Sstd1a:    
                st.info(" The standard deviation is a measure of how spread out the values are from the mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.")

            st.subheader(f"Insight Statistics for 'Y' ({dependent_column_name})")
            st.write("\n")   
            
            if meanY > medianY:
                st.write(f' The mean is higher than the median, which suggests that the data is skewed to the right.')
            elif meanY < medianY:
                st.write(f' The mean is lower than the median, which suggests that the data is skewed to the left.')
            else:
                st.write(f' The mean is equal to the median, which suggests that the data is symmetrical.')

            if stdY > 1:
                st.markdown(
                    f"<span style='color: #344a80;'> The standard deviation is low , </span> "
                    f"<span style='color: red;'>(>1)</span> "
                    f"<span style='color: #344a80;'>, which indicates that the data is concentrated. </span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<span style='color: #344a80;'> The standard deviation is low , </span> "
                    f"<span style='color: #803df5;'>(<=1)</span> "
                    f"<span style='color: #344a80;'>, which indicates that the data is concentrated. </span>",
                    unsafe_allow_html=True
                )

            if meanY > (3 * stdY):
                st.markdown(
                    f"<span style='color: #344a80;'> The difference between the mean is greater than 3 times the standard deviation, </span> "
                    f"<span style='color: red;'>(Mean: {meanY:.3f}, UCL:{meanY + (3 * stdY):.3f}, LCL:{meanY - (3 * stdY):.3f})</span> "
                    f"<span style='color: #344a80;'>, which suggests that there might be significant outliers in the data. </span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<span style='color: #344a80;'> The difference between the mean is less than or equal to 3 times the standard deviation, </span> "
                    f"<span style='color: #803df5;'>(Mean: {meanY:.3f}, UCL:{meanY + (3 * stdY):.3f}, LCL:{meanY - (3 * stdY):.3f})</span> "
                    f"<span style='color: #344a80;'>, which suggests that the data falls within the expected range based on control limits. </span>",
                    unsafe_allow_html=True
                )
                        
            colored_header(
            label="",
            description="",
            color_name="violet-70",
            )          
            st.subheader("Box Plot Graph")
            # st.write("A box plot, shows how a set of data is distributed visually. To visualize the distribution of numerical data and spot trends, patterns, and outliers, statistical analysis frequently uses this technique. The box represents range between the first quartile (25th percentile) and the third quartile, or the interquartile range (IQR), is shown by the box in the box plot (75th percentile). The first quartile is the percentage that divides the lowest 25% of the data from the remaining data, and the third quartile, the highest 25% of the data. The median, which is the number separating the lowest 50% from the highest 50% of the data, is the horizontal line inside the box. It gives the data a measure of central tendency and serves as a reliable guide to the typical value in the data set. The whiskers extend from the box to represent the range of the data, excluding outliers. Outliers are values that are significantly different from the rest of the data and are represented as individual dots outside the whiskers. They can represent errors in data collection, measurement, or data entry and are important to identify as they can skew the results of a statistical analysis.")
            # explanation = """
            # - Is a powerful visualization tool used in statistical analysis to display the distribution of numerical data, allowing for the identification of trends, patterns, and outliers.
            # - The box in a box plot represents the interquartile range (IQR), which provides insights into the spread and variability of the data. It encompasses the middle 50% of the data, with the box's lower edge indicating the 25th percentile (first quartile) and the upper edge representing the 75th percentile (third quartile).
            # - The median, depicted as a horizontal line within the box, illustrates the central tendency of the data. It divides the distribution into two equal halves, with 50% of the data falling below and 50% above this value.
            # - The whiskers extending from the box present the range of the data, excluding outliers. Typically, they extend to a length of 1.5 times the IQR from the edges of the box.
            # - Outliers, indicated as individual data points beyond the whiskers, are values that significantly deviate from the rest of the dataset. They can be caused by various factors such as measurement errors, data entry mistakes, or genuine extreme observations. Identifying and examining outliers is crucial as they can impact the results of statistical analysis and provide valuable insights into data quality, data collection methods, or potential anomalies in the underlying process.
            # - Offers a concise visual summary of the data, enabling researchers to compare distributions, detect skewness, assess symmetry, and make informed decisions based on the data's characteristics.
            # """
            # st.markdown(explanation)            

            with st.expander("More information about Box Plot Graph"):
                reminderA, reminderB, reminderC = st.columns((2.5,2.5,2.5), gap="small")
                with reminderA:
                    st.markdown("<b>1️⃣ Visual summary of the data:</b>", unsafe_allow_html=True)
                    st.info("Is a powerful visualization tool used in statistical analysis to display the distribution of numerical data, allowing for the identification of trends, patterns, and outliers. Enabling researchers to compare distributions, detect skewness, assess symmetry, and make informed decisions based on the data's characteristics.")
                with reminderB:    
                    st.markdown("<b>2️⃣ The box :</b>", unsafe_allow_html=True)
                    st.info("The box in a box plot represents the interquartile range (IQR), which provides insights into the spread and variability of the data. It encompasses the middle 50% of the data, with the box's lower edge indicating the 25th percentile (first quartile) and the upper edge representing the 75th percentile (third quartile).")
                with reminderC:
                    st.markdown("<b>3️⃣ The median:</b>", unsafe_allow_html=True)
                    st.info("The median, depicted as a horizontal line within the box, illustrates the central tendency of the data. It divides the distribution into two equal halves, with 50% of the data falling below and 50% above this value.")

                reminderD, reminderE  = st.columns((2.5,5), gap="small")
                with reminderD:
                    st.markdown("<b>4️⃣ The whiskers:</b>", unsafe_allow_html=True)
                    st.info("The whiskers extending from the box present the range of the data, excluding outliers. Typically, they extend to a length of 1.5 times the IQR from the edges of the box.")
                with reminderE:
                    st.markdown("<b>5️⃣ The Outliers:</b>", unsafe_allow_html=True)
                    st.info("Outliers, indicated as individual data points beyond the whiskers, are values that significantly deviate from the rest of the dataset. They can be caused by various factors such as measurement errors, data entry mistakes, or genuine extreme observations. Identifying and examining outliers is crucial as they can impact the results of statistical analysis and provide valuable insights into data quality, data collection methods, or potential anomalies in the underlying process.")

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

            st.markdown(f"<span style='color: #344a80;'>The x-axis of the box plot represents the independent variable </span><span style='color: #803df5;'>({independent_column_name})</span><span style='color: #344a80;'> which is the variable that you want to examine the relationship between it and the dependent variable. The y-axis represents the dependent variable<span style='color: #803df5;'>({dependent_column_name})</span><span style='color: #344a80;'> which is the variable you want to analyze its distribution.</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: #344a80;'>In conclusion, the box plot provides a quick and effective way to visualize the distribution of the dependent variable </span><span style='color: #803df5;'>({dependent_column_name})</span><span style='color: #344a80;'> across different values of the independent variable</span><span style='color: #803df5;'>({independent_column_name})</span><span style='color: #344a80;'> . By examining the heights and ranges of the boxes, you can gain insights into the relationship between the dependent variable and the independent variable and identify trends, patterns, and outliers in the data.</span>", unsafe_allow_html=True)

    except (UnboundLocalError, ValueError):
        st.error(f'❌ SELECTION ERROR: {dependent_column_name} column might contain categorical/string variables, please select a quantitative column.')
