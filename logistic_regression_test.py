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
    st.header("Logistic Regression")

    with st.expander("What is Logistic Regression?",expanded=True):   
        
        st.write("Binary logistic regression: is used to predict if something will happen or not. It's like guessing if it will rain tomorrow or not based on the weather today.")
        st.markdown("- y: is binary, meaning it can only take on two possible values, such as 0 or 1, yes or no, pass or fail")
        st.markdown("- x: can be continuous or categorical with more than 2 unique values (e.g. height and weight).")
        st.write("")
        st.markdown("For example, you can use binary logistic regression to predict whether a student will pass a test (y) based on their study hours (x).")               
        st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            </style>
            ''', unsafe_allow_html=True)

    if file is not None:

        file_name = file.name

        column_names = data.columns.tolist()
        y_col = column

        # Remove y_col from column_names
        column_names.remove(y_col)
        
        filtered_column_names = []
        for col in column_names:
            if col != y_col:
                if data[col].dtype == np.int64 or data[col].dtype == np.float64:
                    if data[col].nunique() > 2:
                        filtered_column_names.append(col)
                elif data[col].dtype == object:
                    if data[col].nunique() > 2 or is_ordinal(data[col]):
                        filtered_column_names.append(col)
                elif pd.api.types.is_categorical_dtype(data[col]):
                    if data[col].nunique() > 2:
                        filtered_column_names.append(col)
        x_col = st.sidebar.selectbox("4️⃣ SELECT THE 'x' FIELD (independent variable):", [""] + filtered_column_names)
        
        if y_col == x_col:
            st.error("❌ Both columns are the same. Please select different columns.")
            
        else:           
            
            try:
                if x_col == "":
                    st.write("")
                elif ((not pd.api.types.is_string_dtype(data[x_col]) and not pd.api.types.is_integer_dtype(data[x_col])) and data[x_col].nunique() < 2 and not pd.api.types.is_float_dtype(data[x_col]) and not np.issubdtype(data[x_col], np.number)):
                    st.error(f'❌ SELECTION ERROR #1: {x_col} column must be continuous/categorical and discrete data with atleast 2 unique values.')
                elif (data[y_col].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #2: {y_col} column must be binary values.')
                elif ((pd.api.types.is_string_dtype(data[y_col]) and pd.api.types.is_integer_dtype(data[y_col])) and data[y_col].nunique() < 2 and pd.api.types.is_float_dtype(data[y_col]) and np.issubdtype(data[y_col], np.number)):
                    st.error(f'❌ SELECTION ERROR #3: {y_col} column must be binary values.')      
                elif (data[x_col].nunique() < 2):
                    st.error(f'❌ SELECTION ERROR #4: {x_col} column must be continuous/categorical and discrete data with atleast 2 unique values.')     
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
                        recommended_method = ""
                        data[x_col] = data[x_col].values.reshape(-1, 1) 

                    if recommended_method in ["One-Hot", "Ordinal", "Label"]:
                        method = recommended_method
                        transformed_col = transform_column(data, x_col, method)
                        data[x_col] = transformed_col
                    else:
                        selected_cols = st.sidebar.multiselect("COLUMN TO BE NORMALIZED (for selected 'y' field above):", needs_normalization, default=default_values)
                        data = data.copy()
      
                        if len(selected_cols) > 0:
                            method = "Z-Score"
                            numeric_selected_cols = [col for col in selected_cols if col in numeric_cols]
                            categorical_selected_cols = [col for col in selected_cols if col not in numeric_cols]
                            df_norm = normalize_numpy(data, numeric_selected_cols, categorical_selected_cols, method)
                            not_selected_cols = data.columns.difference(selected_cols)
                            data = pd.concat([df_norm, data[not_selected_cols]], axis=1)
                    
                    st.subheader("Table Preview:")
                    st.dataframe(data, height = 400)
                    button, log_row, log_col = st.columns((0.0001,1.5,4.5), gap="small")
                    rows = data.shape[0]
                    cols = data.shape[1]
                    with log_row:
                        st.markdown(f"<span style='color: #803df5;'>➕ Number of rows : </span> <span style='color: #803df5;'>{rows}</span>", unsafe_allow_html=True)
                    with log_col:
                        st.markdown(f"<span style='color: #803df5;'>➕ Number of columns : </span> <span style='color: #803df5;'>{cols}</span>", unsafe_allow_html=True)
                    with button:
                        st.write("")
                    
                    X = data[x_col].to_numpy()
                    y = data[y_col].to_numpy()

                    slope, intercept, r_value, p_value, std_err = linregress(X, y)    
                    
                    if len(X.shape) == 1:
                        X = X.reshape(-1, 1)
                    if len(y.shape) == 1:
                        y = y.reshape(-1, 1).ravel()

                    #test_size = st.sidebar.slider('5️⃣ CHOOSE LOGISTIC TEST SIZE:', 0.1, 0.5, 0.2)

                    training_size = 1 - 0.2

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0) #,random_state=0

                    mean_y = np.mean(y_train)

                    median_y = np.median(y_train)

                    mode_y = data[y_col].mode()[0]

                    std_y = np.std(y_train)

                    training_set_size = X_train.shape[0]

                    test_set_size = X_test.shape[0]

                    # st.sidebar.markdown(f"<span style='color: violet;'>➕ Training set size: </span> <span style='color: #344a80;'>{training_set_size}</span>", unsafe_allow_html=True)
                    # st.sidebar.markdown(f"<span style='color: violet;'>➕ Test set size: </span> <span style='color: #344a80;'>{test_set_size}</span>", unsafe_allow_html=True)

                    model = LogisticRegression()
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    accuracy = model.score(X_test, y_test)
                    cm = confusion_matrix(y_test, y_pred)
                    
                    precision = precision_score(y_test, y_pred, zero_division=1)
                    
                    f1 = f1_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
  
                    colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )  
                    st.subheader("Logistic Regression")

                    st.write("")
 
                    with st.expander("Understanding the Significance Level and P-value",expanded=False):   
                        st.write("The p-value represents the probability that the differences between the groups are due to chance. A small p-value (usually less than 0.05) indicates that the differences between the groups are unlikely to be due to chance, and we can reject the null hypothesis that there is no difference between the groups. In other words, if the p-value is small, it suggests that there is a significant difference between at least two of the groups.")
                        st.write("")
                        st.write("The significance level works in a similar way as in other statistical tests. We set a significance level, usually at 0.05, which represents the maximum probability of making a Type I error, which is rejecting the null hypothesis when it's actually true. If the p-value is less than the significance level, we reject the null hypothesis and conclude that there is a statistically significant difference between at least two of the groups.")
                        st.write("")
                        st.write("To give an analogy, imagine you are comparing the exam scores of three different classes to see if there is a significant difference between them. The null hypothesis would be that there is no significant difference in exam scores between the three classes, and the alternative hypothesis would be that there is a significant difference between at least two of the classes.")
                        st.write("")
                        st.write("You would conduct an ANOVA test and obtain a p-value of 0.02. This means that there is a 2% chance of observing the differences in exam scores between the three classes due to chance. Since the p-value is less than the significance level of 0.05, we reject the null hypothesis and conclude that there is a statistically significant difference in exam scores between at least two of the classes. This information could be useful in identifying areas for improvement or to make decisions about which class may need additional resources or attention.")

                    pvalue1a, pvalue2a = st.columns((2,5), gap="small")
                    with pvalue1a:
                        st.metric("P-value",f"{p_value:.3f}")       

                    with pvalue2a:    
                        st.metric("Significance level:",f"{100-p_value:.3f}")

                    if p_value < 0.05:
                        st.success(f' With p-value of {p_value:.3f} that is less than 0.05, it means that the results for the relationship between {x_col} and {y_col} are statistically significant.')
                        st.write("There is a less than 5% chance that the observed difference between the groups happened due to natural differences alone. This provides strong evidence that the independent variable has a significant impact on the dependent variable.")
                    else:
                        st.warning(f' With p-value of {p_value:.3f} that is greater than 0.05, it means that the results for the relationship between {x_col} and {y_col} are not statistically significant.')
                        st.write("There is a greater than or equal to 5% chance that the observed difference between the groups happened due to natural differences alone. This suggests that the independent variable does not have a significant impact on the dependent variable.")   

                    st.write("\n")
                    st.subheader("Hypothesis Testing")
                    st.write("\n")
                    if p_value <= 0.05:
                        result = "Reject Null Hypothesis"
                        conclusion = "There is sufficient evidence to suggest that {} is a factor on {}.".format(x_col, y_col)
                        st.success("P Value is {:.3f} which is less than or equal to 0.05 ({}); {}".format(p_value, result, conclusion))
                    else:
                        result = "Fail to Reject Null Hypothesis"
                        conclusion = "There is not sufficient evidence to suggest that {} is a factor on {}.".format(x_col, y_col)
                        st.warning("P Value is {:.3f} which is greater than to 0.05 ({}); {}".format(p_value, result, conclusion))

                    null_hypothesis = "The independent variable {} has no effect on the dependent variable {}.".format(x_col, y_col)
                    alternate_hypothesis = "The independent variable {} has an effect on the dependent variable {}.".format(x_col, y_col)
                    st.write("\n\n")
                    st.markdown(f"<span style='color: #803df5; font-weight: bold;'>Null Hypothesis (H0): </span> <span style='color: #344a80;'>{null_hypothesis}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #803df5; font-weight: bold;'>Alternate Hypothesis (H1): </span> <span style='color: #344a80;'>{alternate_hypothesis}</span>", unsafe_allow_html=True)

                    st.write("\n\n")
                    st.markdown(f"<span style='color: #344a80;'>If the p-value is less than or equal to 0.05, it means that the result is statistically significant and we reject the null hypothesis. This suggests that the independent variable </span> <span style='color: #803df5;'>({x_col})</span> <span style='color: #344a80;'> has an effect on the dependent variable </span> <span style='color: #803df5;'>({y_col})</span>. <span style='color: #344a80;'>On the other hand, if the p-value is greater than 0.05, it means that the result is not statistically significant and we fail to reject the null hypothesis. This suggests that the independent variable </span><span style='color: #803df5;'>({x_col})</span> <span style='color: #344a80;'>not have an effect on the dependent variable </span> <span style='color: #803df5;'>({y_col})</span><span style='color: #344a80;'>.</span>", unsafe_allow_html=True)

                    colored_header(
                    label="",
                    description="",
                    color_name="violet-70",
                    )  

                    st.write("\n")

                    st.subheader(f"Descriptive Statistics for 'Y' ({y_col})")
                    st.write("\n")

                    mean, median, mode, std_dev = st.columns((2.5,2.5,2.5,2.5), gap="small")
                    with mean:
                        st.metric("Mean:",f"{mean_y:.3f}")
                    with median:
                        st.metric("Median:",f"{median_y:.3f}")
                    with mode:
                        st.metric("Mode:",f"{mode_y:.3f}")
                    with std_dev:
                        st.metric("Standard Deviation:",f"{std_y:.3f}")
                    
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
                    
                    st.subheader(f"Insight Statistics for 'Y' ({y_col})")
                    st.write("\n")
                    if mean_y > median_y:
                        st.write(f' The mean is higher than the median, which suggests that the data is skewed to the right.')
                    elif mean_y > median_y:
                        st.write(f' The mean is lower than the median, which suggests that the data is skewed to the left.')
                    else:
                        st.write(f' The mean is equal to the median, which suggests that the data is symmetrical.')

                    if std_y > 1:
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

                    if mean_y > (3 * std_y):
                        st.markdown(
                            f"<span style='color: #344a80;'> The difference between the mean is greater than 3 times the standard deviation, </span> "
                            f"<span style='color: red;'>(Mean: {mean_y:.3f}, UCL:{mean_y + (3 * std_y):.3f}, LCL:{mean_y - (3 * std_y):.3f})</span> "
                            f"<span style='color: #344a80;'>, which suggests that there might be significant outliers in the data. </span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<span style='color: #344a80;'> The difference between the mean is less than or equal to 3 times the standard deviation, </span> "
                            f"<span style='color: #803df5;'>(Mean: {mean_y:.3f}, UCL:{mean_y + (3 * std_y):.3f}, LCL:{mean_y - (3 * std_y):.3f})</span> "
                            f"<span style='color: #344a80;'>, which suggests that the data falls within the expected range based on control limits. </span>",
                            unsafe_allow_html=True
                        )


            except TypeError:
                st.error(f'❌ A [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
            except ValueError:
                 st.error(f'❌ B [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
            except AttributeError:
                st.error(f'❌ C [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
