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

def logistic_regression(data, file):
    # Select only numeric columns
    data = data.select_dtypes(include=['object','float','int'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("📝 Logistic Regression Test")

    # Check if a file was uploaded
    if file is not None:
        # Extract the file name from the UploadedFile object
        file_name = file.name
            
        # Display the contents in a scrollable table
        st.dataframe(data)
        st.write(f"Number of rows :{data.shape[0]:.0f}")
        st.write(f"Number of columns :{data.shape[1]:.0f}")
        #ag_grid = AgGrid(data, height=300)
        
        column_names = data.columns.tolist()
        # Get the independent and dependent variables from the user
        x_col = st.selectbox('➕ Select the column name for the X (independent/CATEGORICAL/CONTINUOUS/DISCRETE) variable:', column_names)
        y_col = st.selectbox('➕ Select the column name for the y (dependent/BINARY) variable:', column_names)
        
    # Check if the dependent variable is binary

        if y_col == x_col:
            st.error("❌ Both columns are the same. Please select different columns.")
        else:           
            try:
                if ((not pd.api.types.is_string_dtype(data[x_col]) and not pd.api.types.is_integer_dtype(data[x_col])) and data[x_col].nunique() < 2 and not pd.api.types.is_float_dtype(data[x_col]) and not np.issubdtype(data[x_col], np.number)):
                    st.error(f'❌ {x_col} column must be continuous/categorical and discrete data with atleast 2 unique values.')
                elif (data[y_col].nunique() < 2):
                    st.error(f'❌ {y_col} column must be binary values.')
                elif ((pd.api.types.is_string_dtype(data[y_col]) and pd.api.types.is_integer_dtype(data[y_col])) and data[y_col].nunique() < 2 and pd.api.types.is_float_dtype(data[y_col]) and np.issubdtype(data[y_col], np.number)):
                    st.error(f'❌ {y_col} column must be binary values.')      
                elif (data[x_col].nunique() < 2):
                    st.error(f'❌ {x_col} column must be continuous/categorical and discrete data with atleast 2 unique values.')     
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
                    test_size = st.slider('Choose the test size', 0.1, 0.5, 0.2)

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
                    st.subheader("✎ [Logistic Regression Test]")

                    #st.write(f"Accuracy: {accuracy:.2f}")
                    st.metric("Accuracy",f"{accuracy:.2f}")
                    #st.write(f"Accuracy:", accuracy)
                    # Print insights
                    if accuracy > 0.8:
                        st.success(f"* The model is performing well, with an accuracy of more than 80%.")
                    elif accuracy >= 0.6:
                        st.warning(f"* The model is performing decently, with an accuracy of more than 60% but less than 80%.")
                    else:
                        st.error(f"* The model is performing poorly, with an accuracy of less than 60%. An accuracy of less than 60% typically indicates that the model is making accurate predictions for a small proportion of the test instances and there is a significant need for improvement.")

                    #st.write("Confusion Matrix:")
                    st.metric("Confusion Matrix:",f"{cm:.2f}")
                    #st.write(cm)
                    if cm[1,1] > cm[0,0]:
                        st.success(f"* The model is making more true positive predictions than true negative predictions. The model is making more correct predictions than incorrect predictions. This is generally a good thing, as it indicates that the model is able to accurately classify a large proportion of the test instances.")
                    elif cm[1,1] < cm[0,0]:
                        st.warning(f"* The model is making more true negative predictions than true positive predictions. The model is making more incorrect predictions than correct predictions. This is generally not a good thing, as it indicates that the model is having difficulty accurately classifying the test instances.")
                    else:
                        st.info(f"* The model is making an equal number of true positive and true negative predictions. The model is making an equal number of correct and incorrect predictions. This could indicate that the model is performing poorly.")

                    #st.write(f"Precision:", precision)
                    st.metric("Precision:",f"{precision:.2f}")
                    if precision > 0.8:
                        st.success(f"* The model has a high precision, with fewer false positive predictions.")
                    elif precision > 0.6:
                        st.warning(f"* The model has a decent precision. The model is making fewer false positive predictions than the average model, but it is still making a relatively high number of false positive predictions (more than 20% of the total positive predictions). This could indicate that there is still room for improvement in the model's precision.")
                    else:
                        st.error(f"* The model has a low precision. The model is making a high number of false positive predictions (more than 40% of the total positive predictions). This is generally not a good thing, as it indicates that the model is having difficulty accurately predicting the positive class.")

                    #st.write(f"Recall:", recall)
                    st.metric("Recall:",f"{recall:.2f}")
                    if recall > 0.8:
                        st.success(f"* The model has a high recall, with fewer false negative predictions.")
                    elif recall > 0.6:
                        st.warning(f"* The model has a decent recall. The model is making fewer false negative predictions than the average model, but it is still making a relatively high number of false negative predictions (more than 20% of the total positive instances). This could indicate that there is still room for improvement in the model's recall.")
                    else:
                        st.error(f"* The model has a low recall. The model is making a high number of false negative predictions (more than 40% of the total positive instances). This is generally not a good thing, as it indicates that the model is having difficulty correctly identifying all the positive instances in the test set.")

                    #st.write(f"F1 score:", f1)
                    st.metric("F1 score:",f"{f1:.2f}")
                    if f1 > 0.8:
                        st.success(f"* Insight: The model has a high F1 score, with a balance between precision and recall.")
                    elif f1 > 0.6:
                        st.warning(f"* The model has a decent F1 score, with a balance between precision and recall but room for improvement.")
                    else:
                        st.error(f"* The model has a low F1 score, with a poor balance between precision and recall. The model is having difficulty accurately classifying the test instances.")
            
            except TypeError:
                st.error(f'❌ {x_col} column needs to be categorical/discrete with at least 2 unique values while {y_col} column needs to be in binary values.')  
            except ValueError:
                st.error(f'❌ {x_col} column needs to be categorical/discrete with at least 2 unique values while {y_col} column needs to be in binary values.')  
            except AttributeError:
                st.error(f'❌ {x_col} column needs to be categorical/discrete with at least 2 unique values while {y_col} column needs to be in binary values.')  
