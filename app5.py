import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    
    predict_class = Predictor(model_path='dreaddit/model.joblib', test_data_path = "raw_data/dreaddit-test.csv")
    
    df1 = predict_class.cleaned_output_df.sort_values(by=['residual'],ascending=True).head(3)[['text','y_true','y_pred','residual']]
    
    st.table(df1)
    
    st.latex(r'''
            residual = | predict\_proba - y\_true |
            ''')
