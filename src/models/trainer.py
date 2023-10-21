import pandas as pd
from src.data.preprocess_data import pipeline_build
from src.models.metrics import metrics_calculate
import streamlit as st
from src.visualization.visualization import pred_visualize
import os
from joblib import dump

class Trainer:
    def __init__(self, X, y, external_y, external_final_data, fold_list, horizon, num_cols, cat_cols, frequency, alg,
                 saved_model_path):
        """
        Initialize the Trainer class.

        Parameters:
        - X: DataFrame, input features for training.
        - y: DataFrame, target variable for training.
        - external_y: DataFrame, target variable for external validation.
        - external_final_data: DataFrame, external validation features.
        - fold_list: list, list of train-validation indices for cross-validation.
        - horizon: int, number of time steps to forecast.
        - num_cols: list, numeric columns in the dataset.
        - cat_cols: list, categorical columns in the dataset.
        - frequency: int, time frequency of the data.
        - alg: machine learning algorithm for time series forecasting.
        - saved_model_path: str, path to save the trained models.
        """
        self.X = X
        self.y = y
        self.external_y = external_y
        self.external_final_data = external_final_data
        self.fold_list = fold_list
        self.horizon = horizon
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.frequency = frequency
        self.alg = alg
        self.saved_model_path = saved_model_path

    def train_and_visualization(self):
        """
        Train the time series forecasting model, save it, and visualize predictions and scores.

        Returns:
        - None
        """
        for i in range(len(self.fold_list) + 1):
            train_indices = self.fold_list[0][i]['train']
            val_indices = self.fold_list[0][i]['validation']
            X_train = self.X.iloc[train_indices]
            y_train = self.y.iloc[train_indices]
            X_val = self.X.iloc[val_indices]
            y_val = self.y.iloc[val_indices]
            pipe = pipeline_build(self.alg, self.num_cols, self.cat_cols)
            pipe.fit(X_train, y_train)
            directory = os.path.join(self.saved_model_path)
            if not os.path.exists(os.path.join(directory, str(f'{type(self.alg).__name__}'))):
                os.makedirs(os.path.join(directory, str(f'{type(self.alg).__name__}')), exist_ok=True)
            y_pred = pipe.predict(X_val)
            external_y_pred = pipe.predict(self.external_final_data)
            scores = metrics_calculate(y_val, y_pred, X_train)
            external_scores = metrics_calculate(self.external_y, external_y_pred, self.external_y)
            print(f"Fold {i + 1} Scores : {scores}")
            print(f"Fold {i + 1} External Scores : {external_scores}")
            model_name = os.path.join(f'{directory}/{str(type(self.alg).__name__)}/{str(i)}.gz')
            dump(self.alg, model_name, compress=('gzip', 3))
            model_preds_columns_list = [[f'+{i + 1}_Horizon_time_step'][0] for i in range(self.horizon)]
            y_pred = pd.DataFrame(y_pred, index=X_val.index,
                                  columns=[model_preds_columns_list])
            external_y_pred = pd.DataFrame(external_y_pred, index=self.external_final_data.index,
                                           columns=[model_preds_columns_list])
            """y_pred_ = y_pred.reset_index()
            external_y_pred_ = external_y_pred.reset_index()
            y_pred_ = y_pred_.to_numpy().tolist()
            external_y_pred_ = external_y_pred_.to_numpy().tolist()
            real_results = []
            pred_results = []
            for k, row in enumerate(y_pred_):
                for distance, value in enumerate(row[1:]):
                    real_results.append({
                        "Timestamp": y_pred_[k][0] + timedelta(**{"seconds": (distance + 1) * self.frequency}),
                        "Forecast Distance": distance + 1,
                        "Forecast Point": y_pred_[k][0],
                        "Prediction": value})
            y_pred_ = pd.DataFrame(real_results)
            for k, row in enumerate(external_y_pred_):
                for distance, value in enumerate(row[1:]):
                    pred_results.append({
                        "Timestamp": external_y_pred_[k][0] + timedelta(**{"seconds": (distance + 1) * self.frequency}),
                        "Forecast Distance": distance + 1,
                        "Forecast Point": external_y_pred_[k][0],
                        "Prediction": value})
            external_y_pred_ = pd.DataFrame(pred_results)"""
            st.write(f"**Fold : {i + 1}**")
            st.write(f"Train Start-End: {X_train.index[0]} - {X_train.index[-1]}")
            st.write(f"Validation Start-End: {X_val.index[0]} - {X_val.index[-1]}")
            st.write("Train Scores", scores)
            st.write("External Data Scores", external_scores)
            for k in range(len(y_val.columns.tolist())):
                pred_visualize(y_val, y_pred, k, '', streamlit=True)
                pred_visualize(self.external_y, external_y_pred, k, 'External', streamlit=True)
