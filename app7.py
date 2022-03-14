import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

from dreaddit.predict import Predictor


def app():
    image_outro = Image.open('images/GeorgeEPBox.jpeg')

    
    col1, col2, col3 = st.columns([5,20,2])
    with col1:
        st.write(' ')
    with col2:
        st.image(image_outro)
    with col3:
        st.write(' ')
