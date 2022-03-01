import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from data import get_data, clean_data

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC




MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "dreaddit"

class Trainer(object):

    def __init__(self, X_train, y_train, X_test, y_test):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME


    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    #TODO
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        model = SVC(degree=2, gamma='scale', kernel='poly', coef0=5, probability=True)

        self.pipeline = Pipeline([('SVC', model)])

    def run(self):

        # Set up and fit the pipeline on training data
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

        #add some mflow log info
        # self.mlflow_log_param("model", self.pipeline.get_params())


    def evaluate(self):
        """evaluates the pipeline on test data and return the accuracy"""

        accuracy = self.pipeline.score(self.X_test, self.y_test)
        # self.mlflow_log_metric("accuracy", accuracy)

        return round(accuracy, 4)

# MLFlow methods DO NOT TOUCH!!
    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(
    #             self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
#######

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
    trainer = Trainer(X_train=X_train,
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test)

    # trainer.set_experiment_name('xp2')

    trainer.run()

    accuracy = trainer.evaluate()

    print(f"accuracy: {accuracy}")


    # trainer.save_model()
