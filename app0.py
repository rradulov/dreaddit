import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():
    

    
    st.markdown("<h1 style='text-align: center; color: black;'>Dreaddit Stress Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>Stressed? Find it out by Social Media texts!</h2>", unsafe_allow_html=True)
        

    image_stress = Image.open('images/stressed_person.jpg')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(image_stress)
    with col3:
        st.write(' ')
        
    st.markdown("<h6 style='text-align: center; color: grey;'>By : Helene Lipp, Jack Morrissey, Jeff Doran, Radul Radulov</h6>", unsafe_allow_html=True)
       



