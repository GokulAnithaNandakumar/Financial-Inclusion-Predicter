# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, classification_report
#
# from scipy.stats import pearsonr
# import seaborn as sns
#
# # Load your dataset
# df = pd.read_csv('FINDEXData.csv')
#
# # Assuming 'FinancialInclusion' is your binary target variable
# df['FinancialInclusion'] = (df['2014'] > df['2011']).astype(int)
#
# # Select relevant features
# features = ['2011', '2014', 'MRV']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[features], df['FinancialInclusion'], test_size=0.2, random_state=42)
#
# # Impute missing values
# imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
# X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
#
# # Create and train a Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_imputed, y_train)
#
# # Make predictions on the test set
# predictions = clf.predict(X_test_imputed)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# report = classification_report(y_test, predictions)
#
#
# print(f'Accuracy: {accuracy}')
# print('Classification Report:')
# print(report)
#
# # Assuming 'MRV' is a numerical column
# mean_mrv = df['MRV'].mean()
# a=input("Enter the values in the year 2011: ")
# b=input("Enter the values in the year 2014: ")
# c=input("Enter the new MRV Value ")
# # Assuming 'new_data' has the same features as your training data
# new_data = pd.DataFrame({
#     '2011': [a],  # Replace with the actual value for '2011' for the new region
#     '2014': [b],  # Replace with the actual value for '2014' for the new region60
#
#     'MRV': [c]  # Replace missing values with the mean of 'MRV'
#     # add other features here
# })
#
#
# # Use the trained model to make predictions on the new data
# new_predictions = clf.predict(new_data)
#
# # Display the predictions
# print(f'Predictions for new data: {new_predictions}')

#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
#
# # Load your dataset
# df = pd.read_csv('FINDEXData.csv')
#
# # Select relevant features
# features = ['2011', '2014', 'MRV']
#
# # Drop rows with NaN values in the target variable
# df = df.dropna(subset=['MRV'])
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[features], df['MRV'], test_size=0.2, random_state=42)
#
# # Impute missing values in the features
# imputer = SimpleImputer(strategy='mean')
# X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
#
# # Create and train a Random Forest regressor
# regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# regressor.fit(X_train_imputed, y_train)
#
# # Make predictions on the test set
# predictions = regressor.predict(X_test_imputed)
#
# # Create a DataFrame to hold the actual vs predicted values
# result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
#
# # Plot the regression graph with the median line
# sns.lmplot(x='Actual', y='Predicted', data=result_df, line_kws={'color': 'red'})
# plt.title('Regression Plot with Median Line')
# plt.show()
#
# # Assuming 'new_data' has the same features as your training data
# new_data = pd.DataFrame({
#     '2011': [10000000],  # Replace with the actual value for '2011' for the new region
#     '2014': [10000001],  # Replace with the actual value for '2014' for the new region
#     'MRV': [70.5]  # Replace missing values with the mean of 'MRV'
#     # add other features here
# })
#
# # Use the trained model to make predictions on the new data
# new_predictions = regressor.predict(new_data)
#
# # Display the predictions
# print(f'Predictions for new data: {new_predictions}')
#


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, classification_report
#
# # Load your dataset
# df = pd.read_csv('FINDEXData.csv')
#
# # Assuming 'FinancialInclusion' is your binary target variable
# df['FinancialInclusion'] = (df['2014'] > df['2011']).astype(int)
#
# # Select relevant features for classification
# classification_features = ['2011', '2014', 'MRV']
#
# # Drop rows with NaN values in the target variable for classification
# df_classification = df.dropna(subset=['FinancialInclusion'])
#
# # Split the data for classification
# X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
#     df_classification[classification_features], df_classification['FinancialInclusion'],
#     test_size=0.2, random_state=42
# )
#
# # Impute missing values in classification features
# imputer_cls = SimpleImputer(strategy='mean')
# X_train_cls_imputed = pd.DataFrame(imputer_cls.fit_transform(X_train_cls), columns=X_train_cls.columns)
# X_test_cls_imputed = pd.DataFrame(imputer_cls.transform(X_test_cls), columns=X_test_cls.columns)
#
# # Create and train a Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_cls_imputed, y_train_cls)
#
# # Make predictions on the test set for classification
# predictions_cls = clf.predict(X_test_cls_imputed)
#
# # Evaluate the classification model
# accuracy_cls = accuracy_score(y_test_cls, predictions_cls)
# report_cls = classification_report(y_test_cls, predictions_cls)
#
# print(f'Classification Accuracy: {accuracy_cls*100}%')
# print('Classification Report:')
# print(report_cls)
#
# # Assuming 'MRV' is a numerical column for regression
# regression_features = ['2011', '2014', 'MRV']
#
# # Drop rows with NaN values in the target variable for regression
# df_regression = df.dropna(subset=['MRV'])
#
# # Split the data for regression
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
#     df_regression[regression_features], df_regression['MRV'],
#     test_size=0.2, random_state=42
# )
#
# # Impute missing values in regression features
# imputer_reg = SimpleImputer(strategy='mean')
# X_train_reg_imputed = pd.DataFrame(imputer_reg.fit_transform(X_train_reg), columns=X_train_reg.columns)
# X_test_reg_imputed = pd.DataFrame(imputer_reg.transform(X_test_reg), columns=X_test_reg.columns)
#
# # Create and train a Random Forest regressor
# regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# regressor.fit(X_train_reg_imputed, y_train_reg)
#
# # Make predictions on the test set for regression
# predictions_reg = regressor.predict(X_test_reg_imputed)
#
# # Create a DataFrame to hold the actual vs predicted values for regression
# result_df_reg = pd.DataFrame({'Actual': y_test_reg, 'Predicted': predictions_reg})
#
# # Plot the regression graph with the median line
# sns.lmplot(x='Actual', y='Predicted', data=result_df_reg, line_kws={'color': 'red'})
# plt.title('Regression Plot with Median Line')
#
# # Adjust layout
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
# # Assuming 'new_data' has the same features as your training data for regression
# new_data_reg = pd.DataFrame({
#     '2011': [10000000],  # Replace with the actual value for '2011' for the new region
#     '2014': [10000001],  # Replace with the actual value for '2014' for the new region
#     'MRV': [70.5]  # Replace missing values with the mean of 'MRV'
#     # add other features here
# })
#
# # Use the trained model to make predictions on the new data for regression
# new_predictions_reg = regressor.predict(new_data_reg)
#
# # Display the predictions for regression
# print(f'Predictions for new data (Regression): {new_predictions_reg}')
#


# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
#
# # Load your dataset
# df = pd.read_csv('FINDEXData.csv')
#
# # Assuming 'FinancialInclusion' is your binary target variable
# df['FinancialInclusion'] = (df['2014'] > df['2011']).astype(int)
#
# # Select relevant features
# features = ['2011', '2014', 'MRV']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[features], df['FinancialInclusion'], test_size=0.2,
#                                                     random_state=42)
#
# # Impute missing values
# imputer = SimpleImputer(strategy='mean')
# X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
#
# # Create and train a Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_imputed, y_train)
#
# # Streamlit app
# st.title("Financial Inclusion Predictor")
#
# # User input for new data
# a = st.text_input("Enter the values in the year 2011:")
# b = st.text_input("Enter the values in the year 2014:")
# c = st.text_input("Enter the new MRV Value:")
#
# # Make predictions on the new data
# if st.button("Predict"):
#     new_data = pd.DataFrame({
#         '2011': [float(a)],
#         '2014': [float(b)],
#         'MRV': [float(c)]
#     })
#
#     new_prediction = clf.predict(new_data)
#
#     # Display the prediction
#     if new_prediction[0]==0:
#         st.success("Prediction for new data: Not Financialy included")
#     if new_prediction[0] == 1:
#         st.success("Prediction for new data: Financialy included")
#
#
#
#
# target_variable = 'MRV'
#
# # Select relevant features
# features = ['2011', '2014', 'MRV']
#
# # Drop rows with NaN values in the target variable
# df = df.dropna(subset=[target_variable])
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_variable], test_size=0.2, random_state=42)
#
# # Impute missing values in the features
# imputer = SimpleImputer(strategy='mean')
# X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
#
# # Create and train a Random Forest regressor
# regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# regressor.fit(X_train_imputed, y_train)
#
# # Make predictions on the test set
# predictions = regressor.predict(X_test_imputed)
#
# # Create a DataFrame to hold the actual vs predicted values
# result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
#
# # Plot the regression graph
# plt.figure(figsize=(10, 6))
# sns.regplot(x='Actual', y='Predicted', data=result_df, scatter_kws={'alpha':0.5})
# plt.title('Regression Plot')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.show()


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('FINDEXData.csv')

# Assuming 'FinancialInclusion' is your binary target variable
df['FinancialInclusion'] = (df['2014'] > df['2011']).astype(int)

# Select relevant features for classification
classification_features = ['2011', '2014', 'MRV']

# Drop rows with NaN values in the target variable for classification
df_classification = df.dropna(subset=['FinancialInclusion'])

# Split the data for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    df_classification[classification_features], df_classification['FinancialInclusion'],
    test_size=0.2, random_state=42
)

# Impute missing values in classification features
imputer_cls = SimpleImputer(strategy='mean')
X_train_cls_imputed = pd.DataFrame(imputer_cls.fit_transform(X_train_cls), columns=X_train_cls.columns)
X_test_cls_imputed = pd.DataFrame(imputer_cls.transform(X_test_cls), columns=X_test_cls.columns)

# Create and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls_imputed, y_train_cls)

# Streamlit app
st.title("Financial Inclusion Predictor")

# User input for new data
a = st.text_input("Enter the values in the year 2011:")
b = st.text_input("Enter the values in the year 2014:")
c = st.text_input("Enter the new MRV Value:")

# Make predictions on the new data
if st.button("Predict"):
    new_data = pd.DataFrame({
        '2011': [float(a)],
        '2014': [float(b)],
        'MRV': [float(c)]
    })

    new_prediction = clf.predict(new_data)

    # Display the prediction
    if new_prediction[0] == 0:
        st.success("Prediction for new data: Not Financially included")
    if new_prediction[0] == 1:
        st.success("Prediction for new data: Financially included")

# Regression plot
st.title("Regression Plot")

target_variable = 'MRV'

# Select relevant features
features_reg = ['2011', '2014', 'MRV']

# Drop rows with NaN values in the target variable for regression
df_regression = df.dropna(subset=[target_variable])

# Split the data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    df_regression[features_reg], df_regression[target_variable],
    test_size=0.2, random_state=42
)

# Impute missing values in regression features
imputer_reg = SimpleImputer(strategy='mean')
X_train_reg_imputed = pd.DataFrame(imputer_reg.fit_transform(X_train_reg), columns=X_train_reg.columns)
X_test_reg_imputed = pd.DataFrame(imputer_reg.transform(X_test_reg), columns=X_test_reg.columns)

# Create and train a Random Forest regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg_imputed, y_train_reg)

# Make predictions on the test set
predictions_reg = regressor.predict(X_test_reg_imputed)

# Create a DataFrame to hold the actual vs predicted values for regression
result_df_reg = pd.DataFrame({'Actual': y_test_reg, 'Predicted': predictions_reg})


# Plot the regression graph
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='Actual', y='Predicted', data=result_df_reg, scatter_kws={'alpha': 0.5}, ax=ax)
ax.set_title('Regression Plot')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

# Display the regression plot in Streamlit
st.pyplot(fig)