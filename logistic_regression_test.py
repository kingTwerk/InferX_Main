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

from streamlit_extras.colored_header import colored_header
#from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import os
import datetime

def logistic_regression(data, file):
    # Select only numeric columns
    data = data.select_dtypes(include=['object','float','int'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header("Logistic Regression")
    st.subheader("[👁️‍🗨️] Table Preview:")
    # Check if a file was uploaded
    if file is not None:
        # Extract the file name from the UploadedFile object
        file_name = file.name
        st.dataframe(data)
        #ag_grid = AgGrid(
        #data,
        #key='unique_key_1',
        #height=300, 
        #width='100%',
        #data_return_mode=DataReturnMode.FILTERED_AND_SORTED, 
        #update_mode=GridUpdateMode.FILTERING_CHANGED,
        #fit_columns_on_grid_load=True
        #allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        #)
        button, log_row, log_col = st.columns((5,1,1), gap="small")
        rows = data.shape[0]
        cols = data.shape[1]
        with log_row:
            st.markdown(f"<span style='color: blue;'>Rows : </span> <span style='color: black;'>{rows}</span>", unsafe_allow_html=True)
        with log_col:
            st.markdown(f"<span style='color: blue;'>Columns : </span> <span style='color: black;'>{cols}</span>", unsafe_allow_html=True)
        with button:
            if st.button("Download CSV"):
                # Select only numeric columns
                #ag_grid = data.select_dtypes(include=['float'])
                data = data.select_dtypes(include=['object','float','int'])
                # Get current date
                now = datetime.datetime.now()
                date_string = now.strftime("%Y-%m-%d")
                # Set default save location to desktop
                desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
                save_path = os.path.join(desktop, f'logistic_filtered_data_csv_{date_string}.csv')
                # write data to the selected file
                data.to_csv(save_path, index=False)
                st.success(f'File saved successfully to {save_path}!')

        colored_header(
        label="",
        description="",
        color_name="violet-70",
        )  

        column_names = data.columns.tolist()
        # Get the independent and dependent variables from the user
        st.write("\n")
        x_col = st.selectbox('➕ Select the column name for the X (independent/CATEGORICAL/CONTINUOUS/DISCRETE) variable:', column_names)
        st.write("\n")
        y_col = st.selectbox('➕ Select the column name for the y (dependent/BINARY) variable:', column_names)
        st.write("\n")
        #st.write(data[x_col].nunique())
        #st.write(data[y_col].nunique())
    # Check if the dependent variable is binary

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
                    # Preprocess the data as needed
                    # (e.g., handle missing values, encode categorical variables)
                    for col in data.columns:
                        # Check if the column is a categorical variable
                        if data[col].dtype == "object":
                            # Create a label encoder
                            le = LabelEncoder()

                            # Fit the label encoder to the column
                            le.fit(data[col])

                            # Transform the column
                            data[col] = le.transform(data[col])

                    # Convert dataframe to NumPy array
                    X = data[x_col].to_numpy()
                    y = data[y_col].to_numpy()

                    # Check if X and y are 1D arrays
                    if len(X.shape) == 1:
                        # Reshape X and y to 2D arrays
                        X = X.reshape(-1, 1)
                        y = y.reshape(-1, 1)

                    # Ask the user for the test size
                    st.write("\n")
                    test_size = st.slider('➕ Choose the test size:', 0.1, 0.5, 0.2)

                    # Calculate the training size
                    training_size = 1 - test_size

                    # Random state with a slider
                    # seed_value = st.selectbox("Choose Random Seed", ("None", "Custom"))
                    # if seed_value == "Custom":
                    #     random_state = st.slider('Customize the random seed number', 0, 100, 42)
                    # else:
                    #     random_state = None

                    # Split the data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=0) #,random_state=0

                    # st.write(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

                    # Calculate the mean of Y
                    mean_y = np.mean(y_train)

                    # Calculate the median of Y
                    median_y = np.median(y_train)

                    # Calculate the mode of Y
                    mode_y = data[y_col].mode()[0]

                    # Calculate the standard deviation of Y
                    std_y = np.std(y_train)

                    # Get the sample size of the training set
                    training_set_size = X_train.shape[0]

                    # Get the sample size of the test set
                    test_set_size = X_test.shape[0]

                    # Write the sample sizes to the console or to the app 
                    st.write(f"Training set size: {training_set_size}")
                    st.write(f"Test set size: {test_set_size}")

                    # Display the dataframe
                    #st.dataframe(train_test_df)

                    # Train the model
                    model = LogisticRegression()
                    model.fit(X_train, y_train)

                    # Test the model
                    y_pred = model.predict(X_test)

                    # Calculate evaluation metrics
                    accuracy = model.score(X_test, y_test)
                    cm = confusion_matrix(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)

                    # Print evaluation metrics
                    st.write("\n")
                    st.subheader("[✍] Logistic Regression Test")
                    st.write("\n")
                    #st.write(f"Accuracy: {accuracy:.2f}")
                    Accuracy1a, Accuracy2a = st.columns((1,5), gap="small")
                    with Accuracy1a:
                        st.metric("Accuracy",f"{accuracy:.2f}")
                    #st.write(f"Accuracy:", accuracy)
                    with Accuracy2a:
                        if accuracy > 0.8:
                            st.success(f"* The model is performing well, with an accuracy of more than 80%.")
                        elif accuracy >= 0.6:
                            st.warning(f"* The model is performing decently, with an accuracy of more than 60% but less than 80%.")
                        else:
                            st.error(f"* The model is performing poorly, with an accuracy of less than 60%. An accuracy of less than 60% typically indicates that the model is making accurate predictions for a small proportion of the test instances and there is a significant need for improvement.")
                    st.write("Confusion Matrix:")
                    Matrix1a, Matrix2a = st.columns((1,5), gap="small")
                    with Matrix1a:
                        #st.metric("Confusion Matrix:",f"{cm:.2f}")
                        #st.metric("Confusion Matrix:", cm)
                        st.write(cm)
                    with Matrix2a:
                        if cm[1,1] > cm[0,0]:
                            st.success(f"* The model is making more true positive predictions than true negative predictions. The model is making more correct predictions than incorrect predictions. This is generally a good thing, as it indicates that the model is able to accurately classify a large proportion of the test instances.")
                        elif cm[1,1] < cm[0,0]:
                            st.warning(f"* The model is making more true negative predictions than true positive predictions. The model is making more incorrect predictions than correct predictions. This is generally not a good thing, as it indicates that the model is having difficulty accurately classifying the test instances.")
                        else:
                            st.info(f"* The model is making an equal number of true positive and true negative predictions. The model is making an equal number of correct and incorrect predictions. This could indicate that the model is performing poorly.")
                    Precision1a, Precision2a = st.columns((1,5), gap="small")
                    with Precision1a:
                        #st.write(f"Precision:", precision)
                        st.metric("Precision:",f"{precision:.2f}")
                    with Precision2a:    
                        if precision > 0.8:
                            st.success(f"* The model has a high precision, with fewer false positive predictions.")
                        elif precision > 0.6:
                            st.warning(f"* The model has a decent precision. The model is making fewer false positive predictions than the average model, but it is still making a relatively high number of false positive predictions (more than 20% of the total positive predictions). This could indicate that there is still room for improvement in the model's precision.")
                        else:
                            st.error(f"* The model has a low precision. The model is making a high number of false positive predictions (more than 40% of the total positive predictions). This is generally not a good thing, as it indicates that the model is having difficulty accurately predicting the positive class.")

                    Recall1a, Recall2a = st.columns((1,5), gap="small")
                    with Recall1a:
                        #st.write(f"Recall:", recall)
                        st.metric("Recall:",f"{recall:.2f}")
                    with Recall2a:
                        if recall > 0.8:
                            st.success(f"* The model has a high recall, with fewer false negative predictions.")
                        elif recall > 0.6:
                            st.warning(f"* The model has a decent recall. The model is making fewer false negative predictions than the average model, but it is still making a relatively high number of false negative predictions (more than 20% of the total positive instances). This could indicate that there is still room for improvement in the model's recall.")
                        else:
                            st.error(f"* The model has a low recall. The model is making a high number of false negative predictions (more than 40% of the total positive instances). This is generally not a good thing, as it indicates that the model is having difficulty correctly identifying all the positive instances in the test set.")

                    f1_1a, f1_2a = st.columns((1,5), gap="small")
                    with f1_1a:
                        #st.write(f"F1 score:", f1)
                        st.metric("F1 score:",f"{f1:.2f}")
                    with f1_2a:    
                        if f1 > 0.8:
                            st.success(f"* Insight: The model has a high F1 score, with a balance between precision and recall.")
                        elif f1 > 0.6:
                            st.warning(f"* The model has a decent F1 score, with a balance between precision and recall but room for improvement.")
                        else:
                            st.error(f"* The model has a low F1 score, with a poor balance between precision and recall. The model is having difficulty accurately classifying the test instances.")
                
                    st.write("\n")
                    st.subheader("[📝] Descriptive Statistics for Y")
                    st.write("\n")
                    #st.write(f'Mean: {mean_y:.2f}')
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
                        #st.warning(f'* The difference between the mean and median is greater than 3 times the standard deviation, which suggests that there are outliers in the data.')
                        st.warning(f'* The difference between the mean is greater than 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}) which suggests that there are outliers in the data.')
                    else:
                        #st.info(f'* The difference between the mean and median is less than or equal to 3 times the standard deviation, which suggests that there are no significant outliers in the data.')
                        st.info(f'* The difference between the mean is less than or equal to 3 times the standard deviation, (Mean: {mean_y:.2f}, UCL:{mean_y + (3 * std_y):.2f}, LCL:{mean_y - (3 * std_y):.2f}), which suggests that there are no significant outliers in the data.')
                    

            except TypeError:
                st.error(f'❌ A [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
            except ValueError:
                st.error(f'❌ B [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
            except AttributeError:
                st.error(f'❌ C [{x_col}] column needs to be categorical/discrete with at least 2 unique values while [{y_col}] column needs to be in binary values.')  
