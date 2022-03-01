from logging.handlers import TimedRotatingFileHandler
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from data import get_data, clean_data


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "dreaddit"

class Trainer(object):

    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None

        self.X_train, self.y_train, self.X_test, self.y_test = get_data()

        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME


    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name



    if __name__ == "__main__":
        # Get and clean data
        training_data, test_data = get_data()

        training_data = clean_data(training_data)
        test_data = clean_data(test_data)

        y_train = training_data['label']
        X_train = training_data.drop("label", axis=1)

        y_test = test_data['label']
        X_test = test_data.drop('label', axis=1)


        # # Train and save model, locally and
        # trainer = Trainer(X=X_train, y=y_train)

        # trainer.set_experiment_name('xp2')
        # trainer.run()

        # rmse = trainer.evaluate(X_test, y_test)

        # print(f"rmse: {rmse}")

        # trainer.save_model()
