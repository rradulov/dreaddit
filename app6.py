import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    
    st.write("###")
    
    st.markdown("<h3 style='text-align: center; padding:0px; color: black;'>Good Predictions: stressed</h3>", unsafe_allow_html=True)
    
    st.write("###")

    #predict_class = Predictor(model_path='dreaddit/model.joblib', test_data_path = "raw_data/dreaddit-test.csv")
    
    st.write(""" ### *I just don't know what's real anymore. I can't live with everyone in my life thinking that I'm crazy AND a hysterical ----. I just can't do this anymore. I'm so ashamed I can't be in this skin anymore. I'm starting to get scared.*""")
    
    
    # df2 = predict_class.cleaned_output_df.sort_values(by=['y_true','residual'],ascending=[False, True]).head(3)[['text']]

    # st.table(df2)
    
    st.markdown("""
                
                #### VECTORIZER (negative words) : **crazy**, **hysterical**, **ashamed**, **scared**
                
                """)