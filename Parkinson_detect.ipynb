import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/kaggle/input/parkinson-disease-detection/Parkinsson disease.csv")

df.info()

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

columns = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                          'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                          'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                          'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE', 'status']


plt.figure(figsize=(75, 68))

df_pair_plot = df[columns]
sns.set(style="ticks")
sns.pairplot(df_pair_plot, hue="status", palette={0: 'blue', 1: 'red'})

plt.show()

df_analysis = df[columns]
sns.set(style="whitegrid")

# histograms for each variable
df_analysis.hist(bins=20, figsize=(15, 15))
plt.suptitle('Histogram Analysis Visualization', x=0.5, y=0.95, ha='center', fontsize='x-large')
plt.show()

# correlation heatmap
correl = df.drop(columns=['name', 'status']).corr()
plt.figure(figsize=(20,20))
sns.heatmap(correl,annot=True,cmap='RdYlGn')
plt.show()

numeric_columns = df.select_dtypes(include='number')
skewness = numeric_columns.skew()
print("Skewness for each numeric column:")
print(skewness)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df.drop(['status', 'name'], axis=1)  # Features
y = df['status']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
fg=sns.heatmap(cm,annot=True,cmap="Reds")
figure=fg.get_figure()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Output Confusion Matrix");

# checking prediction vs actual
pd.DataFrame({'actual':y_test,'predict':y_pred})

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

importances_rf = model_rf.feature_importances_

plt.figure(figsize=(12, 8))
plt.bar(X.columns, importances_rf)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Random Forest Feature Importances')
plt.xticks(rotation=45, ha='right')
plt.show()

model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train, y_train)

importances_xgb = model_xgb.feature_importances_

plt.figure(figsize=(12, 8))
plt.bar(X.columns, importances_xgb)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('XGBoost Feature Importances')
plt.xticks(rotation=45, ha='right')
plt.show()

#  ML Pipelines and Model Deployment

!pip install joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

joblib.dump(pipeline, 'parkinson_model.joblib')

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

plt.show()

bar = {'Yes':23, 'No': 8}
diag = list(bar.keys())
values = list(bar.values())
  
fig = plt.figure(figsize = (7, 5))
 
plt.bar(diag, values, align = 'center', color ='maroon',
        width = 0.6)
 
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
plt.title("Diagnosing Parkinson's Disease")
plt.show()


!pip install pyngrok

from pyngrok import ngrok
ngrok.set_auth_token("2bWnoLBUOOrrw5bISwhow40lVjX_5yqLxnpueD4HKRkfVhJiC")

from flask import Flask, render_template, request
from pyngrok import ngrok
def predict_parkinsons(features):
    if np.random.rand() > 0.5:
        return 'Yes'
    else:
        return 'No'
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        features = [float(request.form['feature1']), float(request.form['feature2']), ...]

        diagnosis = predict_parkinsons(features)

        bar = {'Yes': 23, 'No': 8}
        diag = list(bar.keys())
        values = list(bar.values())

        fig = plt.figure(figsize=(7, 5))
        plt.bar(diag, values, align='center', color='maroon', width=0.6)
        plt.xlabel("Diagnosis")
        plt.ylabel("Number of Patients")
        plt.title("Diagnosing Parkinson's Disease")
        plt.savefig('static/diagnosis_plot.png')  # Save the plot to a static file

        return render_template('result.html', diagnosis=diagnosis)

    return render_template('index.html')

if __name__ == '__main__':
    ngrok_url = ngrok.connect(5000)
    print(f' * Running on {ngrok_url}')
    app.run(port=5000)
