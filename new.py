import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from CSV file
file_path = 'FINDEXData.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualization: Boxplot for 2011 and 2014
plt.figure(figsize=(10, 6))
sns.boxplot(x='Indicator Code', y='2011', data=df)
plt.xticks(rotation=90)
plt.title('Boxplot of 2011 Values for Each Indicator')
plt.show()

# Visualization: Distribution of MRV
plt.figure(figsize=(8, 5))
sns.histplot(df['MRV'].dropna(), kde=True, bins=20)
plt.title('Distribution of MRV')
plt.xlabel('MRV')
plt.show()

# Select a subset of indicators for better visualization
indicators_to_plot = df['Indicator Code'].unique()[:10]  # Adjust the number as needed

# Filter the dataframe for the selected indicators
df_subset = df[df['Indicator Code'].isin(indicators_to_plot)]

# Visualization: Violin plot for 2011 values
plt.figure(figsize=(12, 8))
sns.violinplot(x='Indicator Code', y='2011', data=df_subset)
plt.xticks(rotation=45, ha='right')
plt.title('Violin Plot of 2011 Values for Selected Indicators')
plt.show()

# Correlation Matrix
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df[numeric_columns], diag_kind='kde')
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# Countplot
plt.figure(figsize=(10, 6))
sns.countplot(x='Indicator Name', data=df)
plt.xticks(rotation=45)
plt.title('Countplot of Categorical Column')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='2011', y='2014', data=df[numeric_columns])
plt.title('Scatter Plot of 2011 vs 2014')
plt.show()

# Bar Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Indicator Name', y='2011', data=df)
plt.xticks(rotation=45)
plt.title('Bar Plot of 2011 by Category')
plt.show()
