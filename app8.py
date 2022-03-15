import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():

    st.markdown("<h3 style='text-align: center; color: black;'>Natural Language Processing</h3>", unsafe_allow_html=True)
      
    image_nlp = Image.open('images/NLP.png')
    st.image(image_nlp, caption='', use_column_width=True)