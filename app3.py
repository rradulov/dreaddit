import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():

    st.markdown("<h6 style='text-align: center; padding:0px; color: black;'>Most impactful model features</h6>", unsafe_allow_html=True)
    
    image_vars = Image.open('images/var_boxplots1.png')
    st.image(image_vars, caption='', use_column_width=True)

    st.markdown("<h6 style='text-align: center; padding:0px; color: black;'>Confusion Matrix</h6>", unsafe_allow_html=True)
   
    image_conf_matrix = Image.open('images/conf_matrix.png')
    
    col1, mid, col2 = st.columns([20,10,40])
    with col1:
        st.image(image_conf_matrix)
    with col2:
        st.write('This is a confusion matrix')
