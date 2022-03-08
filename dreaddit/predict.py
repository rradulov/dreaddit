import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline


PATH_TO_LOCAL_MODEL = 'model.joblib'

class Predictor(object):

    def __init__(self):

        self.pipeline = self.get_model(PATH_TO_LOCAL_MODEL)
        self.load_data()
        self.predict()
        self.cleaned_output_df = self.output_df[self.output_df.confidence != 0]

    def get_model(self, path_to_joblib):
        pipeline = joblib.load(path_to_joblib)
        return pipeline

    def load_data(self):
        test_data = pd.read_csv("../raw_data/dreaddit-test.csv")

        # np.array(df[1].values.tolist())

        self.y_true = np.array(test_data['label'].values.tolist())
        self.predict_data = test_data.drop(columns = ['label'])

    def predict(self):
        self.y_pred = self.pipeline.predict(self.predict_data)
        self.predict_proba = self.pipeline.predict_proba(self.predict_data)[:,1]

        self.output_df = self.predict_data

        #drop rows where confidence = 0
        # self.output_df = self.output_df[self.output_df.confidence != 0]

        self.output_df['predict_proba'] = self.predict_proba
        self.output_df['y_pred'] = self.y_pred
        self.output_df['y_true'] = self.y_true

        list_of_columns = ['text', 'predict_proba', 'y_pred', 'y_true', 'confidence']
        self.output_df = self.output_df[list_of_columns]
        self.output_df['residual'] = abs(self.output_df['y_true'] -
                                         self.output_df['predict_proba'])

if __name__ == "__main__":
    predictor = Predictor()
    print('worked')
