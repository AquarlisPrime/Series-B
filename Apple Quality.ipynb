import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/kaggle/input/apple-quality/apple_quality.csv")

df.info()

df.columns

#checking missing values
df.isnull().sum()

#check duplicate values
df.duplicated().sum()

# fill the null value
df.fillna(0, inplace=True)

df["Quality"].value_counts()
# to get type of quality of apple

# visualization of apple-id and size
plt.figure(figsize=(35, 20))
sns.scatterplot(data=df, x='A_id', y='Size')
plt.title('Scatter Plot of Apple ID vs Size')
plt.show()

#size and weight of apples
plt.figure(figsize=(30, 18))
sns.regplot(data=df, x='Size', y='Weight', line_kws={'color': 'red'})
plt.xlabel('Size')
plt.ylabel('Weight')
plt.title('Regression Plot of Apple Size vs Weight')
plt.show()

# regression line helps show the overall trend or relationship between them

# A_id and Sweetness
plt.figure(figsize=(20, 15))
sns.swarmplot(data=df, x='A_id', y='Sweetness')
plt.xlabel('Apple ID')
plt.ylabel('Sweetness')
plt.title('Swarm Plot of Apple ID vs Sweetness')
plt.show()

# A_id and Cruchiness
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='A_id', y='Crunchiness', color='green')
plt.xlabel('Apple ID')
plt.ylabel('Crunchiness')
plt.title('Violin Plot of Crunchiness for Each Apple ID')
plt.show()

#Pair Plot
# A pair plot is a matrix of scatterplots and histograms used to visualize relationships and distributions among multiple variables in a dataset.
plt.figure(figsize=(20, 13))
numeric_features = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Quality']]
sns.pairplot(numeric_features, hue='Quality', palette='viridis')
plt.show()

# Distribution Plot
# A distribution plot is a graphical representation that displays the distribution of a univariate variable, typically showing the frequency or probability density of different values along the axis.
numeric_features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness']
for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=feature, hue='Quality', element='step', kde=True, common_norm=False)
    plt.title(f'Distribution of {feature} by Quality')
    plt.show()

# Feature Engineering
# prepare the data for training by encoding categorical variables and performing any necessary transformations.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Drop rows with missing values 
df = df.dropna()

#encode the 'Quality' column into numerical labels
from sklearn.preprocessing import LabelEncoder

# Convert 'Quality' column to strings
df['Quality'] = df['Quality'].astype(str)

label_encoder = LabelEncoder()
df['Quality'] = label_encoder.fit_transform(df['Quality'])

# Feature Engineering
X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness']]
y = df['Quality']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Model:")
print(f"Accuracy: {dt_accuracy}")
print("Classification Report:\n", classification_report(y_test, dt_predictions))

# RandomForest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandomForest Model:")
print(f"Accuracy: {rf_accuracy}")
print("Classification Report:\n", classification_report(y_test, rf_predictions))


# Now for the user input data
# replace these values with user input
user_input = pd.DataFrame({
    'Size': [user_size],
    'Weight': [user_weight],
    'Sweetness': [user_sweetness],
    'Crunchiness': [user_crunchiness],
    'Juiciness': [user_juiciness],
    'Ripeness': [user_ripeness]
})

user_predictions = model.predict(user_input)
predicted_quality = label_encoder.inverse_transform(user_predictions)
print(f'Predicted Quality: {predicted_quality[0]}')

# Decision Tree Predictions for User Input
dt_user_predictions = dt_model.predict(user_input)
dt_user_predicted_quality = label_encoder.inverse_transform(dt_user_predictions)
print("Decision Tree Model Prediction for User Input:")
print(f"Predicted Quality: {dt_user_predicted_quality[0]}")

# RandomForest Predictions for User Input
rf_user_predictions = rf_model.predict(user_input)
rf_user_predicted_quality = label_encoder.inverse_transform(rf_user_predictions)
print("\nRandomForest Model Prediction for User Input:")
print(f"Predicted Quality: {rf_user_predicted_quality[0]}")


