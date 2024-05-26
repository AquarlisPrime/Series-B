import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/kaggle/input/al-dataset/Aluminium alloy dataset for supervised learning/al_data.csv")
df.info()
df.columns
#checking missing values
df.isnull().sum()
#check duplicate values
df.duplicated().sum()
duplicates = df[df.duplicated()]
print("Number of duplicate rows:", len(duplicates))
# Remove duplicate rows
df.drop_duplicates(inplace=True)
# Verify that duplicates have been removed
print("Number of duplicate rows after removal:", len(df[df.duplicated()]))
df.shape
df.head()
df.describe()
print(df['Elongation (%)'].unique())
df['Elongation (%)'] = df['Elongation (%)'].replace("Unknown", np.nan).infer_objects(copy=False)
df['Elongation (%)'] = pd.to_numeric(df['Elongation (%)'], errors='coerce')
# Visualize the distribution of the target variable 'Elongation (%)'
plt.figure(figsize=(10, 6))
sns.histplot(df['Elongation (%)'], kde=True)
plt.title('Distribution of Elongation (%)')
plt.xlabel('Elongation (%)')
plt.ylabel('Frequency')
plt.show()
columns_to_encode = ['Processing', 'class']
df_encoded = pd.get_dummies(df, columns=columns_to_encode)
print(df_encoded.head())
columns_to_replace_unknown = ['Elongation (%)', 'Tensile Strength (MPa)', 'Yield Strength (MPa)']
df[columns_to_replace_unknown] = df[columns_to_replace_unknown].replace("Unknown", np.nan)
df[columns_to_replace_unknown] = df[columns_to_replace_unknown].apply(pd.to_numeric, errors='coerce')
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
print(df[non_numeric_columns].head())
# Handled non-numeric values by replacing 'Unknown' with NaN
df.replace("Unknown", np.nan, inplace=True)

# specific columns to numeric
numeric_columns = ['Elongation (%)', 'Tensile Strength (MPa)', 'Yield Strength (MPa)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# One-hot encode categorical columns
categorical_columns = ['Processing', 'class']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(30, 21))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
elements_columns = df.columns[2:27]  # Assuming columns 2 to 26 are elements
for column in elements_columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=column, y='Elongation (%)', data=df)
    plt.title(f'{column} vs. Elongation (%)')
    plt.xlabel(column)
    plt.ylabel('Elongation (%)')
    plt.show()
# Explore class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
sns.pairplot(df[['Elongation (%)', 'Tensile Strength (MPa)', 'Yield Strength (MPa)']])
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()
plt.figure(figsize=(19, 12))
sns.histplot(df['Elongation (%)'].dropna(), kde=True, color='skyblue', label='Elongation (%)')
sns.histplot(df['Tensile Strength (MPa)'].dropna(), kde=True, color='orange', label='Tensile Strength (MPa)')
sns.histplot(df['Yield Strength (MPa)'].dropna(), kde=True, color='green', label='Yield Strength (MPa)')
plt.title('Distribution of Mechanical Properties')
plt.xlabel('Values')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='class', y='Elongation (%)', data=df)
plt.title('Elongation (%) Distribution by Class')
plt.xlabel('Class')
plt.ylabel('Elongation (%)')
plt.show()
plt.figure(figsize=(25, 17))
sns.countplot(x='Processing', data=df)
plt.title('Count of Samples by Processing Type')
plt.xlabel('Processing Type')
plt.ylabel('Count')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df_regression = df.dropna(subset=['Elongation (%)', 'Tensile Strength (MPa)', 'Yield Strength (MPa)'])

# Define features (X) and target variables (y)
X = df_regression.drop(['Elongation (%)', 'Tensile Strength (MPa)', 'Yield Strength (MPa)'], axis=1)
y = df_regression[['Elongation (%)', 'Tensile Strength (MPa)', 'Yield Strength (MPa)']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# predictions on the test set
y_pred = model.predict(X_test)
