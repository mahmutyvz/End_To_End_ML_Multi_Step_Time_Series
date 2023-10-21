# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm
from functools import partial, reduce
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor


def make_train_test_splits(X, y, test_split):
    """
    This function is specifically designed for time series projects and aims to split time-ordered sequences of input
    data and their corresponding numerical targets into training and testing sets.

    Parameters:

    X (array-like): The time-ordered input data to be split.
    y (array-like): The targets corresponding to the input data.
    test_split (float): The proportion of the data to be used for testing. Should be a value between 0 and 1.
    Returns:

    X_train (array-like): The training data subset.
    X_test (array-like): The testing data subset.
    y_train (array-like): The training targets subset.
    y_test (array-like): The testing targets subset.
    """
    split_size = int(len(X) * (1 - test_split))
    X_train = X[:split_size]
    y_train = y[:split_size]
    X_test = X[split_size:]
    y_test = y[split_size:]
    return X_train, X_test, y_train, y_test


def time_control_type(data, col):
    """
    This function is designed to ensure that a specified column in a DataFrame is of the datetime type.
     If the column is not already in datetime format, it converts the data to datetime.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the target column.
    col (str): The name of the column to be checked and converted to datetime if necessary.
    Returns:

    data (pandas.DataFrame): The DataFrame with the specified column in datetime format.
    """
    if str(data[col].dtypes) not in '[ns]':
        data[col] = pd.to_datetime(data[col])
    return data


def date_sort(data, col):
    """
    This function is designed to sort a DataFrame based on a specified date column in ascending order.

    Parameters:

    data (pandas.DataFrame): The DataFrame to be sorted.
    col (str): The name of the date column based on which the DataFrame should be sorted.
    Returns:

    data (pandas.DataFrame): The sorted DataFrame based on the specified date column in ascending order.
    """
    data = data.sort_values(by=col)
    return data


def frequency_detect(data, selected_datetime):
    """
    This function is designed to analyze the temporal frequencies present in a datetime column of a DataFrame and
    determine the dominant time unit and corresponding frequency.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the datetime column for frequency analysis.
    selected_datetime (str): The name of the datetime column to be analyzed.
    Returns:

    time_type (str): The dominant time unit (e.g., 'years', 'quarters', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds').
    frequency (int): The corresponding frequency of the dominant time unit.
    """
    dates = data[selected_datetime].unique()
    frequencies = [int(int((dates[x + 1] - dates[x])) / (1000000000)) for x in range(0, len(dates) - 1)]
    time_type = ''
    frequency = list(Counter(frequencies).most_common(1)[0])[0]
    if frequency >= 31536000:
        time_type = 'years' if frequency % 31536000 == 0 else 'quarters'
    elif frequency >= 7948800:
        time_type = 'quarters' if frequency % 7948800 == 0 else 'months'
    elif frequency >= 2592000:
        time_type = 'months' if frequency % 2592000 == 0 else 'weeks'
    elif frequency >= 604800:
        time_type = 'weeks' if frequency % 604800 == 0 else 'days'
    elif frequency >= 86400:
        time_type = 'days' if frequency % 86400 == 0 else 'hours'
    elif frequency >= 3600:
        time_type = 'hours' if frequency % 3600 == 0 else 'minutes'
    elif frequency >= 60:
        time_type = 'minutes' if frequency % 60 == 0 else 'seconds'
    elif frequency >= 1:
        time_type = 'seconds'
    return time_type, frequency


def ADF_Test(data, target, selected_datetime_feature, SignificanceLevel=.05):
    """
    This function performs the Augmented Dickey-Fuller (ADF) test on a time series dataset to assess its stationarity.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the time series data.
    target (str): The name of the column representing the time series variable to be tested for stationarity.
    selected_datetime_feature (str): The name of the datetime feature used as an index for time series analysis.
    SignificanceLevel (float, optional): The significance level for the test. Defaults to 0.05.
    Returns:

    isStationary_adf (bool): True if the time series is stationary; False otherwise.
    """
    data = data.set_index(
        pd.DatetimeIndex(data[selected_datetime_feature]))
    data = data.drop([selected_datetime_feature], axis=1)
    data = data[[target]]
    adfTest = adfuller(data, autolag='AIC')

    pValue = adfTest[1]

    if (pValue < SignificanceLevel):
        isStationary_adf = True
    else:
        isStationary_adf = False

    dataResults = pd.Series(adfTest[0:4],
                            index=['Adata Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])

    for key, value in adfTest[4].items():
        dataResults['Critical Value (%s)' % key] = value

    print('Augmented Dickey-Fuller Test Results:')
    print(dataResults)
    print(isStationary_adf)
    return isStationary_adf


def KPSS_Test(data, target, selected_datetime_feature, trend, SignificanceLevel=.05):
    """
    This function performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test on a time series dataset to assess its stationarity.
    Regression: This parameter determines the type of regression to be used in calculating the test statistic.
    The KPSS test applies a regression model to examine the stationarity property of the data.
    In this model, a component of the data is predicted, and the test statistic is calculated based on the remaining residuals.
    "c" (constant): This option represents a regression model with a constant component.
    The test examines the stationarity property of the series with this component.
    "ct" (constant and trend): This option represents a regression model that includes both a constant component and a linear trend component.
    The test assesses the stationarity property of the series with these two components. The nlags parameter specifies the number of lags used in the KPSS test.
    This parameter determines how many steps back the regression model used in calculating the test statistic looks.
    The number of lags is a method used to examine the stationarity property of the series.
    If nlags is set to 25, the regression model will be designed to look back 25 steps (25 observations) when calculating the test statistic.
    This means that the test will use the last 25 observations when evaluating the stationarity property of the series.
    Parameters:

    data (pandas.DataFrame): The DataFrame containing the time series data.
    target (str): The name of the column representing the time series variable to be tested for stationarity.
    selected_datetime_feature (str): The name of the datetime feature used as an index for time series analysis.
    trend (str): The type of regression used in calculating the test statistic. Options are "c" for constant, "ct" for constant and trend.
    SignificanceLevel (float, optional): The significance level for the test. Defaults to 0.05.
    Returns:

    isStationary_kpss (bool): True if the time series is stationary; False otherwise.
    """
    data = data.set_index(
        pd.DatetimeIndex(data[selected_datetime_feature]))
    data = data.drop([selected_datetime_feature], axis=1)
    data = data[[target]]
    kpsstest = kpss(data, regression='ct', nlags=trend)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
    pValue = kpsstest[1]
    if (pValue < SignificanceLevel):
        isStationary_kpss = False
    else:
        isStationary_kpss = True
    print(isStationary_kpss)
    return isStationary_kpss


def editing_index(data, col):
    """
    This function is designed to set a specified column as the index of the DataFrame and drop the original column.

    Parameters:

    data (pandas.DataFrame): The DataFrame to be modified.
    col (str): The name of the column to be set as the index.
    Returns:

    data (pandas.DataFrame): The DataFrame with the specified column set as the index and the original column dropped.
    """
    data = data.set_index(pd.DatetimeIndex(data[col]))
    data = data.drop([col], axis=1)
    return data


def derived_lag_features(data, lag_features, lag_bound=5):
    """
    This function generates lagged features for specified columns in a DataFrame.

    Parameters:

    data (pandas.DataFrame): The DataFrame for which lagged features will be created.
    lag_features (list): A list of column names for which lagged features will be generated.
    lag_bound (int, optional): The maximum number of lagged features to be created. Defaults to 5.
    Returns:

    lagged_data (pandas.DataFrame): A DataFrame containing only the generated lagged features.
    """
    for col in lag_features:
        for lag in range(1, int(lag_bound) + 1):
            f_name = f'{col}_lag_{lag}'
            data[f_name] = data[col].shift(int(lag))
    return data.loc[:, data.columns.str.contains('_lag_')]


def app_lag_data(data, WINDOW, derived_lag_features_cols):
    """
    This function applies lagged features to a given dataset for a specified window size.

    Parameters:

    data (pandas.DataFrame): The DataFrame to which lagged features will be applied.
    WINDOW (int): The size of the window used to create lagged features.
    derived_lag_features_cols (list): A list of column names for which lagged features will be generated.
    Returns:

    lagged_data (pandas.DataFrame): A DataFrame containing the applied lagged features.
    """
    data_1 = data.copy()
    data_2 = []
    v = WINDOW + 1
    if v < 7:
        for i in tqdm(range(len(data_1) - int(WINDOW))):
            data_3 = derived_lag_features(data_1[v - int(WINDOW):v + 1], derived_lag_features_cols,
                                          lag_bound=WINDOW).dropna()
            v += 1
            data_2.append(data_3)
    else:
        for i in tqdm(range(len(data_1) - int(WINDOW))):
            data_3 = derived_lag_features(data_1[v - int(WINDOW):v][-6:], derived_lag_features_cols).dropna()
            v += 1
            data_2.append(data_3)
    lagged_data = pd.concat(data_2)
    lagged_data = lagged_data.astype('float32')
    return lagged_data


def derive_features(data, derivation_lagged_cols, win, window_list, time_type, frequency):
    """
    This function is designed to derive statistical features based on rolling windows for specified lagged columns and window sizes.

    Parameters:

    data (pandas.DataFrame): The DataFrame for which features will be derived.
    derivation_lagged_cols (list): A list of column names for which statistical features will be derived.
    win (int): The default window size for feature derivation.
    window_list (list): A list of window sizes for feature derivation.
    time_type (str): The type of time unit used for window sizes (e.g., 'years', 'quarters', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds').
    frequency (int): The frequency of the data collection in the specified time unit.
    Returns:

    data (pandas.DataFrame): The DataFrame with derived statistical features.
    """
    def time_type_detect(time_type):
        """
        This function is designed to convert a specified time unit into its corresponding number of seconds.

        Parameters:

        time_type (str): The type of time unit to be converted (e.g., 'years', 'quarters', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds').
        Returns:

        time_num (int): The number of seconds corresponding to the specified time unit.
        """
        if time_type == 'years':
            time_num = 31536000
        elif time_type == 'quarters':
            time_num = 7948800
        elif time_type == 'months':
            time_num = 2592000
        elif time_type == 'weeks':
            time_num = 604800
        elif time_type == 'days':
            time_num = 86400
        elif time_type == 'hours':
            time_num = 3600
        elif time_type == 'minutes':
            time_num = 60
        elif time_type == 'seconds':
            time_num = 1
        return time_num

    functions = {
        'min': lambda x: x.rolling(window=win, min_periods=1).min(),
        'max': lambda x: x.rolling(window=win, min_periods=1).max(),
        'mean': lambda x: x.rolling(window=win, min_periods=1).mean(),
        'std': lambda x: x.rolling(window=win, min_periods=1).std(),
        'median': lambda x: x.rolling(window=win, min_periods=1).median()
    }
    time_num = time_type_detect(time_type)
    for win in window_list:
        for function_name, function in functions.items():
            for j in derivation_lagged_cols:
                data[f'{j}_stat_{function_name}_{int(win * frequency / time_num)}_{time_type}'] = data[[j]].apply(
                    function)
    return data


def app_derived_data(data, derived_lag_features_cols, WINDOW, window_list, time_type, frequency):
    """
    This function applies derived statistical features to a given dataset based on specified lagged columns, window sizes, and time units.

    Parameters:

    data (pandas.DataFrame): The DataFrame to which derived features will be applied.
    derived_lag_features_cols (list): A list of column names for which derived statistical features will be generated.
    WINDOW (int): The default window size for feature derivation.
    window_list (list): A list of window sizes for feature derivation.
    time_type (str): The type of time unit used for window sizes (e.g., 'years', 'quarters', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds').
    frequency (int): The frequency of the data collection in the specified time unit.
    Returns:

    derived_data (pandas.DataFrame): A DataFrame containing the applied derived statistical features.
    """
    data_1 = data.copy()
    v = int(WINDOW) + 1
    derives = []
    if v < 7:
        for i in tqdm(range(len(data_1) - int(WINDOW))):
            this_derive = \
                derive_features(data_1[v - int(WINDOW):v + 1], derived_lag_features_cols, WINDOW, window_list,
                                time_type,
                                frequency).iloc[-1].to_frame().T
            derives.append(this_derive)
            v += 1
    else:
        for i in tqdm(range(len(data_1) - int(WINDOW))):
            this_derive = \
                derive_features(data_1[v - int(WINDOW):v], derived_lag_features_cols, WINDOW, window_list, time_type,
                                frequency).iloc[-1].to_frame().T
            derives.append(this_derive)
            v += 1
    derived_data = pd.concat(derives)
    derived_data = derived_data.loc[:, derived_data.columns.str.contains('stat_')]
    derived_data = derived_data.astype('float32')
    return derived_data


def app_diff_data(df, window, lagged_data, derived_data, target, time_type):
    """
    This function applies difference features to a given dataset based on lagged and derived features.

    Parameters:

    df (pandas.DataFrame): The DataFrame to which difference features will be applied.
    window (int): The window size used for feature derivation.
    lagged_data (pandas.DataFrame): A DataFrame containing lagged features.
    derived_data (pandas.DataFrame): A DataFrame containing derived statistical features.
    target (str): The name of the target variable for which difference features will be created.
    time_type (str): The type of time unit used for window sizes (e.g., 'years', 'quarters', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds').
    Returns:

    data (pandas.DataFrame): A DataFrame containing the applied difference features.
    """
    data = df.copy()
    lag_data_target_columns = [x for x in lagged_data.columns if target in x]
    derived_diff_inp_column = derived_data[f'{target}_stat_mean_{window}_{time_type}']
    derived_data_target_columns = [x for x in derived_data.columns if
                                   target in x and derived_diff_inp_column.name not in x]
    for i in lag_data_target_columns:
        data[f'{target}_diff_{i.replace(target + "_", "", 1)}'] = data[target] - lagged_data[i]
        data[f'{target}_{i.replace(target + "_", "", 1)}_diff_{derived_diff_inp_column.name}'] = lagged_data[
                                                                                                     i] - derived_diff_inp_column
    for k in derived_data_target_columns:
        data[f'{target}_diff_{k.replace(target + "_", "", 1)}'] = data[target] - derived_data[k]
        data[f'{target}_{k.replace(target + "_", "", 1)}_diff_{derived_diff_inp_column.name}'] = derived_data[
                                                                                                     k] - derived_diff_inp_column
    data = data.loc[:, data.columns.str.contains('diff')]
    return data


def merge_data(data, lagged_data, derived_data, diff_data=None):
    """
    This function merges multiple DataFrames based on their indexes using a left join.

    Parameters:

    data (pandas.DataFrame): The primary DataFrame to which others will be merged.
    lagged_data (pandas.DataFrame): A DataFrame containing lagged features.
    derived_data (pandas.DataFrame): A DataFrame containing derived statistical features.
    diff_data (pandas.DataFrame, optional): A DataFrame containing difference features. Defaults to None.
    Returns:

    final_data (pandas.DataFrame): A DataFrame containing the merged data.
    """
    list_of_datas = [data, lagged_data, derived_data]
    if not diff_data.empty:
        list_of_datas.append(diff_data)
    merge = partial(pd.merge_asof, left_index=True, right_index=True)
    final_data = reduce(merge, list_of_datas)
    return final_data


def trend_removal_log(data, target_list):
    """
    This function performs trend removal by taking the logarithm of the absolute values of specified target columns.
     If negative values are present in the original data, the function adjusts for them before applying the logarithm.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the target columns for trend removal.
    target_list (list): A list of column names representing the target columns.
    Returns:

    data (pandas.DataFrame): The DataFrame with trend-removed columns after applying logarithm.
    """
    negative_values = []
    [[negative_values.append({column: x}) for x in range(len(data)) if data[column][x] < 0] for index, column in
     enumerate(data[target_list])]
    data[target_list] = abs(data[target_list])
    data[target_list] = np.log1p(data[target_list])
    if not negative_values:
        [data.rename({x: f"{x}_log"}, axis=1, inplace=True) for x in data[target_list].columns.tolist()]
        return data
    for i in negative_values:
        for key, value in i.items():
            data[key][value] = data[key][value] * int(-1)
    [data.rename({x: f"{x}_log"}, axis=1, inplace=True) for x in data[target_list].columns.tolist()]
    return data


def split(df, target, horizon):
    """
    This function is designed to split a DataFrame into features (X) and target variable (y) considering a specified prediction horizon.

    Parameters:

    df (pandas.DataFrame): The original DataFrame to be split.
    target (str): The name of the target variable.
    horizon (int): The prediction horizon, representing the number of periods into the future for forecasting.
    Returns:

    X (pandas.DataFrame): The features DataFrame.
    y (pandas.DataFrame): The target variable DataFrame.
    """
    X = df.drop([target], axis=1)
    y = df[[target]]
    horizon = int(horizon)
    y = derived_lag_features(y, [target], horizon).dropna()
    X = X.iloc[horizon:]
    return X, y


def get_fold(X, y, fold_number):
    """
    This function generates time series cross-validation (CV) partitions for a specified number of folds using the TimeSeriesSplit method.

    Parameters:

    X (pandas.DataFrame): The features DataFrame.
    y (pandas.DataFrame): The target variable DataFrame.
    fold_number (int): The number of folds for time series cross-validation.
    Returns:

    cv_partitions (list): A list of dictionaries containing train and validation indices for each CV fold.
    tscv (TimeSeriesSplit): The TimeSeriesSplit object with the specified number of folds.
    """
    tscv = TimeSeriesSplit(n_splits=fold_number)
    tscv.get_n_splits(X, y)
    cv_partitions = []
    for train_index, val_index in tscv.split(X, y):
        cv_partitions.append({f'train': train_index.tolist(),
                              f'validation': val_index.tolist()})
    return cv_partitions, tscv


def pipeline_build(alg, num_cols, cat_cols):
    """
    This function constructs a scikit-learn pipeline for preprocessing numeric and categorical features and applying a specified machine learning algorithm.

    Parameters:

    alg (object): The machine learning algorithm to be used. It should be compatible with scikit-learn's estimator interface.
    num_cols (list): A list of column names representing numeric features.
    cat_cols (list): A list of column names representing categorical features.
    Returns:

    pipe (Pipeline): A scikit-learn pipeline that preprocesses features and applies the specified machine learning algorithm.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', fill_value='missing')),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)], remainder='passthrough')

    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('algorithm', MultiOutputRegressor(alg))])
    return pipe
