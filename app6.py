import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    
    st.markdown("<h3 style='text-align: center; padding:0px; color: black;'>Good Predictions: stressed</h3>", unsafe_allow_html=True)
    
    st.write("###")

    predict_class = Predictor(model_path='dreaddit/model.joblib', test_data_path = "raw_data/dreaddit-test.csv")
    
    df2 = predict_class.cleaned_output_df.sort_values(by=['y_true','residual'],ascending=[False, True]).head(3)[['text']]

    st.table(df2)
    
