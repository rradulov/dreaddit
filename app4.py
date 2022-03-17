import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    
    st.markdown("<h3 style='text-align: center; padding:0px; color: black;'>Model Insights</h3>", unsafe_allow_html=True)
    
    st.write("###")
    
    # predict_class = Predictor(model_path='dreaddit/model.joblib', test_data_path = "raw_data/dreaddit-test.csv")
    
    # df = predict_class.cleaned_output_df.sort_values(by=['confidence','residual'],ascending=False).head(3)[['text','y_true','y_pred']]
    # df = df.rename(columns={'y_true': 'Actual','y_pred':'Predicted'})
    # df['Actual']= df['Actual'].map({1:'stressed', 0:'non-stressed'})
    # df['Predicted']= df['Predicted'].map({1:'stressed', 0:'non-stressed'})
    
    
    st.write(""" ### *Wrote this in about three minutes today. I need to work some things out but it is crazy how much writing has helped me in the past three and a half months. My therapist credits my journaling to the progress I am continuing to make. I want to send my thoughts to everyone posting on this sub, I am strong not only for myself but for you. Being in a toxic relationship is so isolating you don't even realize how many people feel hopeless and desperate for change.*""")
    
    
    
    # st.table(df)

    # st.markdown(""" Where do predictions go south:""")
    # st.markdown("""
    #             - reflective language
    #             - telling stories in 3rd person
    #             """)
    # st.markdown("""**NLP** is a very complex task for a computer to interpret human emotions from text
    #              """)