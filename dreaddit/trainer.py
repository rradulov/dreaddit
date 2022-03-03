import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from data import get_data, clean_data

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "dreaddit"

class Trainer(object):

    def __init__(self, X_train, y_train, X_test, y_test, model):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        #self.pipeline = None
        self.model = model

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

        #model = SVC(degree=2, gamma='scale', kernel='poly', coef0=5, probability=True)

        self.pipeline = Pipeline([('estimator', self.model)])

    def run(self):

        # Set up and fit the pipeline on training data
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

        #add some mflow log info
        self.mlflow_log_param("model", 'voting hard classifier')

        # for key, value in self.pipeline.get_params().items():
        #         self.mlflow_log_param(key, value)



    def evaluate(self):
        """evaluates the pipeline on test data and return the accuracy"""

        accuracy = self.pipeline.score(self.X_test, self.y_test)
        self.mlflow_log_metric("accuracy", accuracy)

        return round(accuracy, 4)

    # MLFlow methods DO NOT TOUCH!!
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
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


    clf1 = SVC(degree=2, gamma='scale', kernel='poly', coef0=5, probability=True)
    clf2 = RandomForestClassifier(max_depth=25, min_samples_leaf=6, n_estimators=250)
    clf3 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70)
    clf4 = MultinomialNB(alpha=5)
    clf5 = GaussianNB()
    clf6 = QuadraticDiscriminantAnalysis()

    estimators = [('svc', clf1), ('rfc', clf2), ('gbc', clf3),
                  ('nb', clf4), ('gnb', clf5), ('qda', clf6)]


    stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    vote_model_hard = VotingClassifier(estimators=estimators, voting='hard')

    #estimators = [clf1, clf2, clf3, clf4, clf5, clf6]
    #for estimator in estimators:

    # # Train and save model, locally and
    trainer = Trainer(X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model = vote_model_hard)


    trainer.run()

    accuracy = trainer.evaluate()

    print(f"accuracy: {accuracy}")


    # trainer.save_model()
