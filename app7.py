import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    st.markdown(""" #### Where do predictions go south:""")
    st.markdown("""
                - ##### reflective language
                - ##### telling stories in 3rd person
                """)
    # st.markdown("""**NLP** is a very complex task for a computer to interpret human emotions from text
    #              """)
    
    image_outro = Image.open('images/GeorgeEPBox.jpeg')

    col1, col2, col3 = st.columns([5,20,2])
    with col1:
        st.write(' ')
    with col2:
        st.image(image_outro)
    with col3:
        st.write(' ')
