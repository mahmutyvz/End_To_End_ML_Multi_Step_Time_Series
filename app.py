import streamlit as st
import datetime
import warnings
import pandas as pd
from paths import Path
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.models.trainer import Trainer
from src.visualization.visualization import date_column_info
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.data.preprocess_data import *
from src.features.feature_engineering import date_engineering
from src.models.hyperparameter_optimize import optuna_optimize

warnings.filterwarnings("ignore")
st.set_page_config(page_title="End_To_End_Multiple_Time_Series_Regression",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("<h1 style='text-align:center;'>Daily Delhi Climate Prediction</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))
tabs = ["Data Analysis", "Visualization", "Train", "About"]
page = st.sidebar.radio("Tabs", tabs)

if page == "Data Analysis":
    df = pd.read_csv(Path.train_path)
    df = df[:-1]
    df = time_control_type(df, Path.timestamp_column)
    df = date_sort(df, Path.timestamp_column)
    variables = {
        "descriptions": {
            "meantemp": "Mean temperature",
            "humidity": "Humidity value for the day (units are grams of water vapor per cubic meter volume of air).",
            "wind_speed": "Wind speed measured in kmph.",
            "meanpressure": "Pressure reading of weather (measure in atm)",
        }
    }
    profile = ProfileReport(df, title="Daily Delhi Climate Prediction", variables=variables, dataset={
        "description": "The Dataset is fully dedicated for the developers who want to train the model on Weather Forecasting for Indian climate."
                       " This dataset provides data from 1st January 2013 to 24th April 2017 in the city of Delhi, India. "
                       "The 4 parameters here are meantemp, humidity, wind_speed, meanpressure.",
        "url": "https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data"})
    st.title("Data Overview")
    st.write(df)
    st_profile_report(profile)

elif page == "Train":
    option = st.radio(
        'What model would you like to use for training?',
        ('XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'))
    if option == 'XGBRegressor':
        model = XGBRegressor(random_state=Path.random_state)
    elif option == 'LGBMRegressor':
        model = LGBMRegressor(random_state=Path.random_state,verbose=-1)
    elif option == 'CatBoostRegressor':
        model = CatBoostRegressor(random_seed=Path.random_state)
    with st.spinner("Training is in progress, please wait..."):
        df = pd.read_csv(Path.train_path)
        external_df = pd.read_csv(Path.test_path)
        df = df[:-1]
        df = time_control_type(df, Path.timestamp_column)
        external_df = time_control_type(external_df, Path.timestamp_column)
        df = date_sort(df, Path.timestamp_column)
        external_df = date_sort(external_df, Path.timestamp_column)
        train = date_engineering(df, Path.timestamp_column)
        test = date_engineering(external_df, Path.timestamp_column)
        time_type, frequency = frequency_detect(train, Path.timestamp_column)
        isStationary_adf = ADF_Test(train, Path.target, Path.timestamp_column)
        isStationary_kpss = KPSS_Test(train, Path.target, Path.timestamp_column, trend=Path.window)
        train = editing_index(train, Path.timestamp_column)
        test = editing_index(test, Path.timestamp_column)
        test_day = test.index[0] - datetime.timedelta(days=Path.window)
        external_data = pd.concat([train[train.index >= test_day], test])
        num_cols = train.select_dtypes(include=['float', 'int']).columns.tolist()
        cat_cols = train.select_dtypes(exclude=['float', 'int']).columns.tolist()
        lagged_data = app_lag_data(train, Path.window, num_cols)
        external_lagged_data = app_lag_data(external_data, Path.window, num_cols)
        derived_data = app_derived_data(train, num_cols, Path.window, Path.window_list, time_type, frequency)
        external_derived_data = app_derived_data(external_data, num_cols, Path.window, Path.window_list, time_type,
                                                 frequency)
        if int(Path.window) < int(6):
            train = train.iloc[int(Path.window) + 1:]
        else:
            train = train.iloc[int(Path.window):]
        if not isStationary_kpss:
            diff_data = app_diff_data(train, Path.window, lagged_data, derived_data, Path.target, time_type)
            external_diff_data = app_diff_data(external_data[Path.window:], Path.window, external_lagged_data,
                                               external_derived_data,
                                               Path.target, time_type)
        final_data = merge_data(train, lagged_data, derived_data, diff_data)
        external_final_data = merge_data(external_data[Path.window:], external_lagged_data, external_derived_data,
                                         external_diff_data)
        if not isStationary_adf:
            target_list = [x for x in final_data.columns.tolist() if x.startswith(Path.target) and x != Path.target]
            final_data = trend_removal_log(final_data, target_list)
            external_final_data = trend_removal_log(external_final_data, target_list)
        X, y = split(final_data, Path.target, Path.horizon)
        external_x, external_y = split(external_data, Path.target, Path.horizon)
        external_y = external_y[external_y.index >= external_final_data.index[0]]
        fold_list = get_fold(X, y, 3)
        num_cols = X.select_dtypes(include=['float', 'int']).columns.tolist()
        cat_cols = X.select_dtypes(exclude=['float', 'int']).columns.tolist()
        best_params, best_value = optuna_optimize(X, y, fold_list, model, num_cols, cat_cols)
        model.set_params(**best_params)
        trainer = Trainer(X, y, external_y, external_final_data, fold_list, Path.horizon, num_cols, cat_cols, frequency,
                          model, Path.models_path)
        trainer.train_and_visualization()

elif page == "Visualization":
    df = pd.read_csv(Path.train_path)
    df = df[:-1]
    with st.spinner("Visuals are being generated, please wait..."):
        df = time_control_type(df, Path.timestamp_column)
        df = date_sort(df, Path.timestamp_column)
        date_column_info(df, streamlit=True)

elif page == "About":
    st.header("Contact Info")
    st.markdown("""**mahmutyvz324@gmail.com**""")
    st.markdown("""**[LinkedIn](https://www.linkedin.com/in/mahmut-yavuz-687742168/)**""")
    st.markdown("""**[Github](https://github.com/mahmutyvz)**""")
    st.markdown("""**[Kaggle](https://www.kaggle.com/mahmutyavuz)**""")
st.set_option('deprecation.showPyplotGlobalUse', False)
