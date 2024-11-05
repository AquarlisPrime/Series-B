import pandas as pd
import numpy as np
data = pd.read_csv('/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv', parse_dates=['date'], index_col='date')
data.dtypes
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats

data = pd.read_csv('/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv', 
                   parse_dates=['date'], index_col='date')

#Data Cleaning
def clean_and_convert(column):
    # Replace % and commas, convert to numeric, and coerce errors
    return pd.to_numeric(column.replace({',': '', '%': ''}, regex=True), errors='coerce')

for col in data.columns:
    data[col] = clean_and_convert(data[col])

# Handling Missing Values
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)

# Outlier Handling
z_scores = np.abs(stats.zscore(data))
data = data[(z_scores < 3).all(axis=1)]

# Feature Engineering
lags = 4  
for lag in range(1, lags + 1):
    data[f'Silica_Concentrate_lag_{lag}'] = data['% Silica Concentrate'].shift(lag)

window_size = 3  
data['rolling_mean'] = data['% Silica Concentrate'].rolling(window=window_size).mean()
data['rolling_std'] = data['% Silica Concentrate'].rolling(window=window_size).std()

data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek

data['pH_Density_interaction'] = data['Ore Pulp pH'] * data['Ore Pulp Density']

data.dropna(inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# Train-Test Split 
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

train_target = train_data['% Silica Concentrate']
train_features = train_data.drop(columns=['% Silica Concentrate'])
test_target = test_data['% Silica Concentrate']
test_features = test_data.drop(columns=['% Silica Concentrate'])

# Visualize
print(train_data.head())
print(f'Train features shape: {train_features.shape}, Train target shape: {train_target.shape}')
print(f'Test features shape: {test_features.shape}, Test target shape: {test_target.shape}')

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D, GlobalAveragePooling1D, concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Func LSTM 
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(128, return_sequences=True)(inputs)  
    lstm_out = LSTM(64, return_sequences=False)(lstm_out)  
    lstm_out = BatchNormalization()(lstm_out)  
    lstm_out = Dropout(0.3)(lstm_out)  
    lstm_out = Dense(64, activation='relu')(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out) 
    return inputs, lstm_out

# Func TCN model
def build_tcn_model(input_shape):
    inputs = Input(shape=input_shape)
    conv_out = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(inputs)  
    conv_out = BatchNormalization()(conv_out) 
    conv_out = Dropout(0.3)(conv_out)  
    conv_out = Conv1D(64, kernel_size=2, activation='relu', padding='causal')(conv_out)  
    conv_out = GlobalAveragePooling1D()(conv_out)
    conv_out = Dense(64, activation='relu')(conv_out)
    conv_out = Dropout(0.2)(conv_out)  
    return inputs, conv_out

timesteps = 10
num_features = train_data.shape[1]  

# Reshape
def reshape_data(data, timesteps):
    X, y_silica, y_iron = [], [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y_silica.append(data[i + timesteps, -2])  
        y_iron.append(data[i + timesteps, -1])    
    return np.array(X), np.array(y_silica), np.array(y_iron)

X_train, y_train_silica, y_train_iron = reshape_data(train_data.to_numpy(), timesteps)

# branch
lstm_inputs, lstm_out = build_lstm_model((timesteps, num_features))
tcn_inputs, tcn_out = build_tcn_model((timesteps, num_features))

merged_out = concatenate([lstm_out, tcn_out])

output_silica = Dense(1, activation='linear', name='silica_output')(merged_out)
output_iron = Dense(1, activation='linear', name='iron_output')(merged_out)

model = Model(inputs=[lstm_inputs, tcn_inputs], outputs=[output_silica, output_iron])

# Compiling
optimizer = Adam(learning_rate=1e-4, clipvalue=1.0) 
model.compile(
    optimizer=optimizer, 
    loss={
        'silica_output': 'mse',  
        'iron_output': 'mse'      
    },
    loss_weights={
        'silica_output': 1.0,    
        'iron_output': 1.0        
    }
)

# Callbacks 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)  
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)  

# Fitting
history = model.fit(
    [X_train, X_train], 
    [y_train_silica, y_train_iron],  
    validation_split=0.2, 
    epochs=30, 
    batch_size=64, 
    callbacks=[reduce_lr, early_stopping, model_checkpoint]
)


print(history.history.keys())  

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # loss
    if 'silica_output_loss' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['silica_output_loss'], label='Silica Output Loss')
    if 'iron_output_loss' in history.history:
        plt.plot(history.history['iron_output_loss'], label='Iron Output Loss')
    plt.title('Output Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Call 
plot_training_history(history)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from flask import Flask
from pyngrok import ngrok

# Flask and Dash
app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app)

# Dash
dash_app.layout = html.Div([
    html.H1("Flotation Process Monitoring Dashboard"),
    dcc.Graph(id='silica-forecast-plot'),
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
])

@dash_app.callback(
    Output('silica-forecast-plot', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_silica_forecast(n_intervals):
    # Simulation
    timesteps = 10
    num_features = 5  # number of features
    new_data = np.random.rand(timesteps, num_features)
    silica_prediction = np.random.rand(timesteps)  

    # figures
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=silica_prediction.flatten(), mode='lines', name='Silica Concentration Forecast'))
    fig.update_layout(title='Silica Concentration Forecast over Time', xaxis_title='Time Steps', yaxis_title='Silica Concentration')
    
    return fig

public_url = ngrok.connect(8050)
print("Dash app is live! Public URL:", public_url)

dash_app.run_server(port=8050)

from sklearn.metrics import mean_absolute_error, r2_score
import shap
import numpy as np
import os

print("Test Data Shape:", test_data.shape)

def reshape_data(data, timesteps):
    X = []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
    return np.array(X)

X_test = reshape_data(test_data.values, timesteps)

predictions = model.predict([X_test, X_test])  

true_silica = test_data.iloc[timesteps-1:, -2].values  
true_iron = test_data.iloc[timesteps-1:, -1].values 

print(f"Shape of true_silica: {true_silica.shape}")

if isinstance(predictions, list):
    predictions_silica = predictions[0][:, 0]  
    predictions_iron = predictions[1][:, 0]  
else:
    predictions_silica = predictions[:, 0]  
    predictions_iron = predictions[:, 1]  

print(f"Shape of predictions_silica before adjustment: {predictions_silica.shape}")
print(f"Shape of predictions_iron before adjustment: {predictions_iron.shape}")

if len(true_silica) != len(predictions_silica):
    print(f"Adjusting predictions_silica from {predictions_silica.shape[0]} to {true_silica.shape[0]}")
    predictions_silica = np.pad(predictions_silica, (0, len(true_silica) - len(predictions_silica)), 'constant', constant_values=0)[:len(true_silica)]

if len(true_iron) != len(predictions_iron):
    print(f"Adjusting predictions_iron from {predictions_iron.shape[0]} to {true_iron.shape[0]}")
    predictions_iron = np.pad(predictions_iron, (0, len(true_iron) - len(predictions_iron)), 'constant', constant_values=0)[:len(true_iron)]

print(f"Shape of predictions_silica after adjustment: {predictions_silica.shape}")
print(f"Shape of predictions_iron after adjustment: {predictions_iron.shape}")

print(f"Last few values of true_silica: {true_silica[-5:]}")
print(f"Last few values of predictions_silica: {predictions_silica[-5:]}")

# Silica
mae_silica = mean_absolute_error(true_silica, predictions_silica)
r2_silica = r2_score(true_silica, predictions_silica)
print(f'Silica MAE: {mae_silica}, R²: {r2_silica}')

# Iron
mae_iron = mean_absolute_error(true_iron, predictions_iron)
r2_iron = r2_score(true_iron, predictions_iron)
print(f'Iron MAE: {mae_iron}, R²: {r2_iron}')

# SHAP
sample_size = 1000  
background_data = X_train[np.random.choice(X_train.shape[0], sample_size, replace=False)]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

try:
    if isinstance(predictions, tuple):
        explainer = shap.DeepExplainer(model, background_data)
        
        test_sample_size = 500  
        X_test_sample = X_test[:test_sample_size]  
        
        shap_values = explainer.shap_values([X_test_sample, X_test_sample]) 
        
        shap.summary_plot(shap_values, X_test_sample)

    else:
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values([X_test_sample, X_test_sample])  
        shap.summary_plot(shap_values, X_test_sample)

except Exception as e:
    print(f"Error initializing SHAP explainer: {e}")
