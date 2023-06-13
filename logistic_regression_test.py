import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from scipy.stats import linregress

from functions import is_ordinal, normalize_numpy, transform_column

from streamlit_extras.colored_header import colored_header

import os
import datetime

def logistic_regression(df_final, file, column):

    data = df_final.select_dtypes(include=['object','float','int'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header("🧮 Logistic Regression")

    with st.expander("What is Logistic Regression?",expanded=True):   
        
        st.write("Binary logistic regression: is used to predict if something will happen or not. It's like guessing if it will rain tomorrow or not based on the weather today.")
        st.markdown("- y: is binary, meaning it can only take on two possible values, such as 0 or 1, yes or no, pass or fail")
        st.markdown("- x: can be continuous or categorical (e.g. height and weight).")
        st.write("")
        st.markdown("For example, you can use binary logistic regression to predict whether a student will pass a test (y) based on their study hours (x).")               
        st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)

    st.subheader("[👁️‍🗨️] Table Preview:")

    if file is not None:

        file_name = file.name

        column_names = data.columns.tolist()
        y_col = column

        filtered_column_names = []
        for col in column_names:
            if col != y_col:
                if data[col].dtype == np.int64 or data[col].dtype == np.float64:
                    if data[col].nunique() > 2:
                        filtered_column_names.append(col)
                elif data[col].dtype == object:
                    if data[col].nunique() > 2 or is_ordinal(data[col]):
                        filtered_column_names.append(col)

        x_col = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", filtered_column_names)
        
        if y_col == x_col:
            st.error("❌ Both columns are the same. Please select different columns.")
            
        else:           
            
            try:
                if ((not pd.api.types.is_string_dtype(data[x_col]) and not pd.api.types.is_integer_dtype(data[x_col])) and data[x_col].nunique() < 2 and not pd.api.types.is_float_dtype(data[x_col]) and not np.issubdtype(data[x_col], np.number)):
                    st.error(f'❌ 1 {x_col} column must be continuous/categorical and discrete data with atleast 2 unique values.')
                elif (data[y_col].nunique() < 2):
                    st.error(f'❌ 2 {y_col} column must be binary values.')
                elif ((pd.api.types.is_string_dtype(data[y_col]) and pd.api.types.is_integer_dtype(data[y_col])) and data[y_col].nunique() < 2 and pd.api.types.is_float_dtype(data[y_col]) and np.issubdtype(data[y_col], np.number)):
                    st.error(f'❌ 3 {y_col} column must be binary values.')      
                elif (data[x_col].nunique() < 2):
                    st.error(f'❌ 4 {x_col} column must be continuous/categorical and discrete data with atleast 2 unique values.')     
                else:

                    label_encoder = LabelEncoder()
                    label_encoder.fit(data[y_col])
                    data[y_col] = label_encoder.transform(data[y_col])
                    
                    numeric_cols = data._get_numeric_data().columns
                    categorical_cols = data.columns.difference(numeric_cols)

                    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

                    needs_normalization = []
                    for col in numeric_cols:
                        z_scores = (data[col] - data[col].mean()) / data[col].std()
                        if (z_scores.max() - z_scores.min()) > 3:
                            needs_normalization.append(col)

                    common_cols = set([x_col]).intersection(set(needs_normalization))

                    if common_cols:
                        default_values = list(common_cols)
                    else:
                        default_values = []
                    
                    x_col_type = None
                    if data[x_col].dtype == np.int64:
                        x_col_type = "integer"
                    elif data[x_col].dtype == np.float64:
                        x_col_type = "float"
                    else:
                        x_col_type = "object"

                    levels = {}
                    if x_col_type == "integer":
                        unique_values = data[x_col].nunique()
                        if unique_values == 2:
                            levels[x_col] = "binary"
                        else:
                            levels[x_col] = "discrete"
                    elif x_col_type == "float":
                        unique_values = data[x_col].nunique()
                        if unique_values == 2:
                            levels[x_col] = "binary"
                        else:
                            levels[x_col] = "continuous"
                    else:
                        unique_values = data[x_col].nunique()
                        if unique_values == 2:
                            levels[x_col] = "binary"
                        else:
                            if is_ordinal(data[x_col]):
                                levels[x_col] = "ordinal"
                            else:
                                levels[x_col] = "nominal"

                    if levels[x_col] == "nominal" and unique_values > 2:
                        recommended_method = "One-Hot"
                    elif levels[x_col] == "ordinal":
                        recommended_method = "Ordinal"
                    elif levels[x_col] == "continuous":
                        recommended_method = "Z-Score"
                    else:
                        data[x_col] = data[x_col].values.reshape(-1, 1) 

                    if recommended_method in ["One-Hot", "Ordinal", "Label"]:
                        method = recommended_method
                        transformed_col = transform_column(data, x_col, method)
                        data[x_col] = transformed_col
                    else:
                        selected_cols = st.sidebar.multiselect("👉 COLUMN TO BE NORMALIZED (for selected 'y' field above):", needs_normalization, default=default_values)
                        data = data.copy()
      
                        if len(selected_cols) > 0:
                            method = "Z-Score"
                            numeric_selected_cols = [col for col in selected_cols if col in numeric_cols]
                            categorical_selected_cols = [col for col in selected_cols if col not in numeric_cols]
                            df_norm = normalize_numpy(data, numeric_selected_cols, categorical_selected_cols, method)
                            not_selected_cols = data.columns.difference(selected_cols)
                            data = pd.concat([df_norm, data[not_selected_cols]], axis=1)
                    
                    st.dataframe(data, height = 400)
                    button, log_row, log_col = st.columns((0.0001,1.5,4.5), gap="small")
                    rows = data.shape[0]
                    cols = data.shape[1]
                    with log_row:
                        st.markdown(f"<span style='color: violet;'>➕ Number of rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
                    with log_col:
                        st.markdown(f"<span style='color: violet;'>➕ Number of columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)
                    with button:
                        st.write("")
                    
                    X = data[x_col].to_numpy()
                    y = data[y_col].to_numpy()

                    slope, intercept, r_value, p_value, std_err = linregress(X, y)    
                    
                    if len(X.shape) == 1:
                        X = X.reshape(-1, 1)
                    if len(y.shape) == 1:
                        y = y.reshape(-1, 1).ravel()

                    test_size = st.sidebar.slider('5️⃣ CHOOSE LOGISTIC TEST SIZE:', 0.1, 0.5, 0.2)

                    training_size = 1 - test_size

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=0) #,random_state=0

                    mean_y = np.mean(y_train)

                    median_y = np.median(y_train)

                    mode_y = data[y_col].mode()[0]

                    std_y = np.std(y_train)

                    training_set_size = X_train.shape[0]

                    test_set_size = X_test.shape[0]

                    st.sidebar.markdown(f"<span style='color: violet;'>➕ Training set size: </span> <span style='color: black;'>{training_set_size}</span>", unsafe_allow_html=True)
                    st.sidebar.markdown(f"<span style='color: violet;'>➕ Test set size: </span> <span style='color: black;'>{test_set_size}</span>", unsafe_allow_html=True)

                    model = LogisticRegression()
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    accuracy = model.score(X_test, y_test)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    precision = precision_score(y_test, y_pred, zero_division=1)
                    
                    f1 = f1_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
  
                    st.markdown("---")  
                    st.subheader("[✍] Logistic Regression")
                    # st.write("\n")
                    # A1, CA1 = st.columns((1,5), gap="small")
                    # with A1:
                    #     st.write("")
                    # with CA1:
                    #     with st.expander("Why Accuracy?",expanded=False):   
                    #         st.write("Accuracy is a measure of how well a model is able to correctly predict something. In the context of logistic regression, accuracy is the percentage of times the model correctly predicted the outcome of a binary (yes or no) event.")
                    #         st.write("")
                    #         st.write("For example, let's say you are trying to predict whether a coin will land heads or tails when you flip it. If you flip the coin 10 times and your model correctly predicts the outcome 8 times, then your accuracy is 80% (8 out of 10).")
                    #         st.write("")
                    #         st.write("An analogy for accuracy could be a basketball player trying to make shots. If the player shoots 10 times and makes 8 of those shots, their accuracy is 80%. In the same way, if a model correctly predicts the outcome of an event 80% of the time, we can say that its accuracy is 80%.")
                            
                    # Accuracy1a, Accuracy2a = st.columns((1,5), gap="small")
                    # with Accuracy1a:
                    #     st.metric("Accuracy",f"{accuracy:.2f}")
                    # with Accuracy2a:
                    #     if accuracy > 0.8:
                    #         st.success(f"* The model is performing well, with an accuracy of more than 80%.")
                    #     elif accuracy >= 0.6:
                    #         st.warning(f"* The model is performing decently, with an accuracy of more than 60% but less than 80%.")
                    #     else:
                    #         st.error(f"* The model is performing poorly, with an accuracy of less than 60%. An accuracy of less than 60% typically indicates that the model is making accurate predictions for a small proportion of the test instances and there is a significant need for improvement.")
                    
                    # st.write("\n")
                    # F1, C1 = st.columns((1,5), gap="small")
                    # with F1:
                    #     st.write("")
                    # with C1:
                    #     with st.expander("Understanding the Precision, Recall and F1 score",expanded=False):   
                    #         st.write("Precision, recall, and F1 score are three important measures of a model's performance")
                    #         st.write("")
                    #         st.write("Precision is a measure of how precise or exact the model's predictions are. It is the proportion of true positive predictions out of all the positive predictions made by the model. An analogy for precision could be a basketball player's accuracy when shooting free throws. If the basketball player takes 10 shots but only intends to make 5 of them, and they make all 5 of those shots, their precision is 100%. In the same way, a model's precision is good if it correctly predicts a high proportion of positive outcomes out of all the positive predictions it made.")
                    #         st.write("")
                    #         st.write("Recall is a measure of a model's ability to correctly identify all of the positive cases in the data, regardless of whether it makes some false positive predictions. It is the proportion of true positive predictions out of all the actual positive cases in the data. An analogy for recall could be a metal detector on a beach looking for hidden treasure. If there are 100 pieces of treasure hidden in the sand, and the metal detector is able to correctly detect 80 of them, its recall is 80%.")
                    #         st.write("")
                    #         st.write("The F1 score is a measure of a model's performance that takes both precision and recall into account. It is the harmonic mean of precision and recall, and it ranges from 0 to 1, with higher values indicating better performance. An analogy for the F1 score could be baking a cake. If you make a cake that is very healthy but not very delicious, your precision may be low (i.e. not very precise in achieving the goal of making a delicious cake), but your recall may be high (i.e. you are able to achieve the goal of making a healthy cake). Conversely, if you make a cake that is very delicious but not very healthy, your precision may be high (i.e. you are precise in achieving the goal of making a delicious cake), but your recall may be low (i.e. you are not achieving the goal of making a healthy cake). The F1 score takes both precision and recall into account and finds a balance between them. A model that balances both precision and recall well will have a high F1 score, indicating good overall performance.")

                    # Precision1a, Precision2a = st.columns((1,5), gap="small")
                    # with Precision1a:
                    #     st.metric("Precision:",f"{precision:.2f}")
                       
                    # with Precision2a:    
                    #     if precision > 0.8:
                    #         st.success(f"* The model has a high precision, with fewer false positive predictions.")
                    #     elif precision > 0.6:
                    #         st.warning(f"* The model has a decent precision. The model is making fewer false positive predictions than the average model, but it is still making a relatively high number of false positive predictions (more than 20% of the total positive predictions). This could indicate that there is still room for improvement in the model's precision.")
                    #     else:
                    #         st.error(f"* The model has a low precision. The model is making a high number of false positive predictions (more than 40% of the total positive predictions). This is generally not a good thing, as it indicates that the model is having difficulty accurately predicting the positive class.")

                    # Recall1a, Recall2a = st.columns((1,5), gap="small")
                    # with Recall1a:
                    #     st.metric("Recall:",f"{recall:.2f}")
                        
                    # with Recall2a:
                    #     if recall > 0.8:
                    #         st.success(f"* The model has a high recall, with fewer false negative predictions.")
                    #     elif recall > 0.6:
                    #         st.warning(f"* The model has a decent recall. The model is making fewer false negative predictions than the average model, but it is still making a relatively high number of false negative predictions (more than 20% of the total positive instances). This could indicate that there is still room for improvement in the model's recall.")
                    #     else:
                    #         st.error(f"* The model has a low recall. The model is making a high number of false negative predictions (more than 40% of the total positive instances). This is generally not a good thing, as it indicates that the model is having difficulty correctly identifying all the positive instances in the test set.")

                    # f1_1a, f1_2a = st.columns((1,5), gap="small")
                    # with f1_1a:
                    #     st.metric("F1 score:",f"{f1:.2f}")
                        
                    # with f1_2a:    
                    #     if f1 > 0.8:
                    #         st.success(f"* Insight: The model has a high F1 score, with a balance between precision and recall.")
                    #     elif f1 > 0.6:
                    #         st.warning(f"* The model has a decent F1 score, with a balance between precision and recall but room for improvement.")
                    #     else:
                    #         st.error(f"* The model has a low F1 score, with a poor balance between precision and recall. The model is having difficulty accurately classifying the test instances.")

                    st.write("")
 
                    with st.expander("Understanding the Significance Level and P-value",expanded=False):   
                        st.write("The p-value in ANOVA represents the probability that the differences between the groups are due to chance. A small p-value (usually less than 0.05) indicates that the differences between the groups are unlikely to be due to chance, and we can reject the null hypothesis that there is no difference between the groups. In other words, if the p-value is small, it suggests that there is a significant difference between at least two of the groups.")
                        st.write("")
                        st.write("The significance level in ANOVA works in a similar way as in other statistical tests. We set a significance level, usually at 0.05, which represents the maximum probability of making a Type I error, which is rejecting the null hypothesis when it's actually true. If the p-value is less than the significance level, we reject the null hypothesis and conclude that there is a statistically significant difference between at least two of the groups.")
                        st.write("")
                        st.write("To give an analogy, imagine you are comparing the exam scores of three different classes to see if there is a significant difference between them. The null hypothesis would be that there is no significant difference in exam scores between the three classes, and the alternative hypothesis would be that there is a significant difference between at least two of the classes.")
                        st.write("")
                        st.write("You would conduct an ANOVA test and obtain a p-value of 0.02. This means that there is a 2% chance of observing the differences in exam scores between the three classes due to chance. Since the p-value is less than the significance level of 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference in exam scores between at least two of the classes. This information could be useful in identifying areas for improvement or to make decisions about which class may need additional resources or attention.")

                    pvalue1a, pvalue2a = st.columns((1,5), gap="small")
                    with pvalue1a:
                        st.metric("P-value",f"{p_value:.2f}")       

                    with pvalue2a:    
                        st.metric("Significance level:",f"{100-p_value:.2f}")

                    if p_value < 0.05:
                        st.success(f' With p-value of {p_value:.2f} that is less than 0.05, it means that the results for the relationship between {x_col} and {y_col} are statistically significant.')
                        st.write("There is a less than 5% chance that the observed difference between the groups happened due to natural differences alone. This provides strong evidence that the independent variable has a significant impact on the dependent variable.")
                    else:
                        st.warning(f' With p-value of {p_value:.2f} that is greater than 0.05, it means that the results for the relationship between {x_col} and {y_col} are not statistically significant.')
                        st.write("There is a greater than or equal to 5% chance that the observed difference between the groups happened due to natural differences alone. This suggests that the independent variable does not have a significant impact on the dependent variable.")   

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

                    st.write("\n")
                    st.subheader("[🧪] Hypothesis Testing")
                    st.write("\n")
                    if p_value <= 0.05:
                        result = "Reject Null Hypothesis"
                        conclusion = "There is sufficient evidence to suggest that {} is a factor on {}.".format(x_col, y_col)
                        st.success("P Value is {:.2f} which is less than or equal to 0.05 ({}); {}".format(p_value, result, conclusion))
                    else:
                        result = "Fail to Reject Null Hypothesis"
                        conclusion = "There is not sufficient evidence to suggest that {} is a factor on {}.".format(x_col, y_col)
                        st.warning("P Value is {:.2f} which is greater than to 0.05 ({}); {}".format(p_value, result, conclusion))

                    null_hypothesis = "The independent variable {} has no effect on the dependent variable {}.".format(x_col, y_col)
                    alternate_hypothesis = "The independent variable {} has an effect on the dependent variable {}.".format(x_col, y_col)
                    st.write("\n\n")
                    st.markdown(f"<span style='color: blue; font-weight: bold;'>Null Hypothesis (H0): </span> <span style='color: black;'>{null_hypothesis}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: blue; font-weight: bold;'>Alternate Hypothesis (H1): </span> <span style='color: black;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

                    st.write("\n\n")
                    st.markdown(f"<span style='color: black;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: blue;'>({x_col})</span> <span style='color: black;'> has an effect on the dependent variable </span> <span style='color: blue;'>({y_col})</span>. <span style='color: black;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: blue;'>({x_col})</span> <span style='color: black;'>not have an effect on the dependent variable </span> <span style='color: blue;'>({y_col})</span><span style='color: black;'>.</span>", unsafe_allow_html=True)

                    colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )  


                    st.write("\n")


                    st.subheader(f"[📝] Descriptive Statistics for 'Y' ({y_col})")
                    st.write("\n")

                    mean, median, mode, std_dev = st.columns((2.5,2.5,2.5,2.5), gap="small")
                    with mean:
                        st.metric("Mean:",f"{mean_y:.2f}")
                    with median:
                        st.metric("Median:",f"{median_y:.2f}")
                    with mode:
                        st.metric("Mode:",f"{mode_y:.2f}")
                    with std_dev:
                        st.metric("Standard Deviation:",f"{std_y:.2f}")
                    
                    meanS, medianS, modeS, std_devS = st.columns((2.5,2.5,2.5,2.5), gap="small")
                    with meanS:
                        st.info(" The mean is the average of all the values in the data set. It is calculated by adding up all the values and then dividing by the total number of values.")
                    with medianS:
                        st.info(" The median is the middle value when the data is arranged in order. It is the value that separates the data into two equal halves.")
                    with modeS:
                        st.info(" The mode is the value that appears most frequently in the data set. A data (set can have one mode, more than one mode, or no mode at all.")
                    with std_devS:
                        st.info(" The standard deviation is a measure of how spread out the values are from the mean. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.")

                    st.write("\n")
                    
                    st.subheader(f"[💡] Insight Statistics for 'Y' ({y_col})")
                    st.write("\n")
                    if mean_y > median_y:
                        st.write(f' The mean is higher than the median, which suggests that the data is skewed to the right.')
                    elif mean_y > median_y:
                        st.write(f' The mean is lower than the median, which suggests that the data is skewed to the left.')
                    else:
                        st.write(f' The mean is equal to the median, which suggests that the data is symmetrical.')

                    if std_y > 1:
                        st.markdown(
                            f"<span style='color: black;'> The standard deviation is low , </span> "
                            f"<span style='color: red;'>(>1)</span> "
                            f"<span style='color: black;'>, which indicates that the data is concentrated. </span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<span style='color: black;'> The standard deviation is low , </span> "
                            f"<span style='color: blue;'>(<=1)</span> "
                            f"<span style='color: black;'>, which indicates that the data is concentrated. </span>",
                            unsafe_allow_html=True
                        )

                    if mean_y > (3 * std_y):
                        st.markdown(
                            f"<span style='color: black;'> The difference between the mean is greater than 3 times the standard deviation, </span> "
                            f"<span style='color: red;'>(Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f})</span> "
                            f"<span style='color: black;'>, which suggests that there might be significant outliers in the data. </span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<span style='color: black;'> The difference between the mean is less than or equal to 3 times the standard deviation, </span> "
                            f"<span style='color: blue;'>(Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f})</span> "
                            f"<span style='color: black;'>, which suggests that the data falls within the expected range based on control limits. </span>",
                            unsafe_allow_html=True
                        )

            except TypeError:
                st.error(f'❌ A [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
            except ValueError:
                 st.error(f'❌ B [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
            except AttributeError:
                st.error(f'❌ C [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
