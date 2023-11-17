import streamlit as st
import pandas as pd
import numpy as np
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

# Visualization: Boxplot for 2011 and 2014
st.write("### Boxplot of 2011 Values for Each Indicator")
boxplot_fig, boxplot_ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Indicator Code', y='2011', data=df, ax=boxplot_ax)
boxplot_ax.set_xticks(rotation=90)
boxplot_ax.set_title('Boxplot of 2011 Values for Each Indicator')
st.pyplot(boxplot_fig)

# Visualization: Distribution of MRV
st.write("### Distribution of MRV")
hist_fig, hist_ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['MRV'].dropna(), kde=True, bins=20, ax=hist_ax)
hist_ax.set_title('Distribution of MRV')
hist_ax.set_xlabel('MRV')
st.pyplot(hist_fig)

# Select a subset of indicators for better visualization
indicators_to_plot = df['Indicator Code'].unique()[:10]  # Adjust the number as needed

# Filter the dataframe for the selected indicators
df_subset = df[df['Indicator Code'].isin(indicators_to_plot)]

# Visualization: Violin plot for 2011 values
st.write("### Violin Plot of 2011 Values for Selected Indicators")
violin_fig, violin_ax = plt.subplots(figsize=(12, 8))
sns.violinplot(x='Indicator Code', y='2011', data=df_subset, ax=violin_ax)
violin_ax.set_xticks(rotation=45, ha='right')
violin_ax.set_title('Violin Plot of 2011 Values for Selected Indicators')
st.pyplot(violin_fig)

# Correlation Matrix
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()
st.write("### Correlation Matrix")
corr_fig, corr_ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=corr_ax)
corr_ax.set_title('Correlation Matrix')
st.pyplot(corr_fig)

# Pairplot
st.write("### Pairplot of Numerical Features")
pairplot_fig = sns.pairplot(df[numeric_columns], diag_kind='kde')
pairplot_fig.fig.suptitle('Pairplot of Numerical Features', y=1.02)
st.pyplot(pairplot_fig)

# Countplot
st.write("### Countplot of Categorical Column")
countplot_fig, countplot_ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='Indicator Name', data=df, ax=countplot_ax)
countplot_ax.set_xticks(rotation=45)
countplot_ax.set_title('Countplot of Categorical Column')
st.pyplot(countplot_fig)

# Scatter Plot
st.write("### Scatter Plot of 2011 vs 2014")
scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='2011', y='2014', data=df[numeric_columns], ax=scatter_ax)
scatter_ax.set_title('Scatter Plot of 2011 vs 2014')
st.pyplot(scatter_fig)

# Bar Plot
st.write("### Bar Plot of 2011 by Category")
barplot_fig, barplot_ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Indicator Name', y='2011', data=df, ax=barplot_ax)
barplot_ax.set_xticks(rotation=45)
barplot_ax.set_title('Bar Plot of 2011 by Category')
st.pyplot(barplot_fig)
