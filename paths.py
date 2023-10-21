class Path:
    """
    The Path class contains configurations and file paths for a time series project, specifically focused on climate data for Delhi.

    Attributes:
        target (str): The target variable for the time series project.
        timestamp_column (str): The column representing timestamps in the data.
        root (str): The root directory for the project.
        train_path (str): The file path to the raw Daily Delhi Climate training data.
        cleaned_train_path (str): The file path to the preprocessed and cleaned training data.
        test_path (str): The file path to the raw Daily Delhi Climate test data.
        cleaned_test_path (str): The file path to the preprocessed and cleaned test data.
        models_path (str): The directory path to store trained models.
        fold_number (int): The number of folds for time series cross-validation.
        hyperparameter_trial_number (int): The number of trials for hyperparameter tuning.
        window (int): The size of the rolling window used in feature engineering.
        window_list (list of int): A list of window sizes for feature engineering.
        horizon (int): The forecast horizon for time series predictions.
        random_state (int): The random seed for reproducibility.
    """
    target = 'meantemp'
    timestamp_column = 'date'
    root = 'C:/Users/MahmutYAVUZ/Desktop/Software/Python/kaggle/multiple_time_series/'
    train_path = root + '/data/raw/DailyDelhiClimateTrain.csv'
    cleaned_train_path = root + '/data/preprocessed/cleaned_train.csv'
    test_path = root + '/data/external/DailyDelhiClimateTest.csv'
    cleaned_test_path = root + '/data/preprocessed/cleaned_test.csv'
    models_path = root + "/models/"
    fold_number = 5
    hyperparameter_trial_number = 3
    window = 365
    window_list = [365, 180, 90]
    horizon = 7
    random_state = 42
