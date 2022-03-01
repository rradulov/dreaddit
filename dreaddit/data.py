import pandas as pd

def get_data():
    """method to get the data from CSVs"""

    training_data = pd.read_csv("../raw_data/model_processed_data.csv")
    test_data = pd.read_csv("../raw_data/model_processed_data_test.csv")

    #print('success')

    return training_data, test_data

def clean_data(df):
    "Clean up the data"

    return df
