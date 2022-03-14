import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    
    predict_class = Predictor(model_path='dreaddit/model.joblib', test_data_path = "raw_data/dreaddit-test.csv")
    
    df = predict_class.cleaned_output_df.sort_values(by=['confidence','residual'],ascending=False).head(3)[['text','y_true','y_pred','residual']]
    
    st.table(df)

    st.markdown(""" Where do predictions go south:""")
    st.markdown("""
                - reflective language
                - telling stories in 3rd person
                """)
    st.markdown("""**NLP** is a very complex task for a computer to interprit humman emotions from text, 
                however we do believe our project is ..... 
                 """)