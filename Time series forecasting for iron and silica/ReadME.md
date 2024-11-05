# Flotation Process Quality Prediction with LSTM-TCN Hybrid Model

This repository is a streamlined approach for predicting quality metrics in the flotation process of a mining plant using a hybrid LSTM-TCN (Long Short-Term Memory - Temporal Convolutional Network) model. The project encompasses end-to-end steps from data preparation and model training to deployment on a live monitoring dashboard.

## Project Overview
The goal of this project is to predict the percentage concentrations of Silica and Iron within a mining process. By leveraging both LSTM and TCN architectures, this model captures the temporal and local patterns essential for time-series analysis in industrial processes. The inclusion of a real-time monitoring dashboard provides actionable insights for continuous process optimization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset contains extensive time-series data from a mining plant's flotation process, with features such as pH, pulp density, and chemical concentrations impacting output quality.

- **Source**: `/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv`

## Pipeline Overview
1. **Data Loading & Cleaning**:
   - Removes special characters (e.g., `%`, `,`).
   - Handles missing values with median imputation.
   - Outliers are filtered using Z-scores for better model accuracy.
2. **Feature Engineering**: 
   - Generates time-based and rolling statistics, interaction terms, and lag features.
3. **Scaling**: 
   - Standardizes features to aid model convergence.
4. **Train-Test Split**: 
   - Divides data for model validation.
5. **Model Building**: 
   - Constructs a hybrid LSTM-TCN model.
6. **Evaluation**: 
   - Assesses model performance using MAE and R² metrics.
7. **Visualization**: 
   - Displays loss and output loss curves for model convergence insights.
8. **Real-Time Monitoring Dashboard**:
   - Provides ongoing process tracking via Dash.

## Installation
Clone this repository and install the required libraries:
```bash
git clone https://github.com/yourusername/flotation-quality-prediction.git
cd flotation-quality-prediction
pip install -r requirements.txt
```

**Required Libraries**:
- `pandas`, `numpy`, `scikit-learn`
- `keras`, `tensorflow`, `matplotlib`
- `dash`, `plotly`, `flask`, `pyngrok`

## Usage
1. **Data Preprocessing**:
   - Load, clean, and engineer features.
2. **Model Training**:
   - Train the LSTM-TCN model on prepared data.
   - Fine-tune with callbacks like `ReduceLROnPlateau` and `EarlyStopping`.
3. **Evaluation**:
   - Evaluate with Mean Absolute Error (MAE) and R² scores.
4. **Dashboard**:
   - Start the dashboard locally:
     ```bash
     python app.py
     ```
   - Use the Ngrok URL to access it remotely.

## Model Architecture
The hybrid LSTM-TCN model architecture includes:
- **LSTM Layers**: For capturing long-term temporal dependencies.
- **TCN Layers**: For extracting local feature patterns.
- **Fully Connected Layers**: For final prediction of Silica and Iron concentrations.

## Evaluation
Performance metrics include:
- **Silica Concentrate Prediction**: MAE and R².
- **Iron Concentrate Prediction**: MAE and R².
- **SHAP Analysis**: To interpret predictions and understand feature contributions.

### Example Evaluation Metrics
| Metric                   | Silica Concentrate | Iron Concentrate |
|--------------------------|--------------------|------------------|
| **Mean Absolute Error**  | 2.15%             | 1.87%           |
| **R² Score**             | 0.92              | 0.89            |

## Visualizations
- **Loss Curves**: Monitor training and validation loss over epochs.
- **Output Loss**: Separate loss tracking for silica and iron predictions.
  
## Dashboard
The real-time monitoring dashboard enables:
- **Silica Concentration Forecast**: Live plot of predicted silica concentration over time.
- **Dynamic Updates**: Refreshes based on latest predictions for proactive monitoring.

## Contributing
We welcome contributions! Fork the repository and submit a pull request with your updates.

## License
This project is licensed under the MIT License. 

This repository provides both a predictive model and a practical, real-time tool for monitoring and improving mining process efficiency, aiding operational decision-making through advanced time-series forecasting and visualization.
