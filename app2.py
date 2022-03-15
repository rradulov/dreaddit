import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():

    st.markdown("<h3 style='text-align: center; color: black;'>Word Clouds</h3>", unsafe_allow_html=True)
    
    st.markdown("<h5 style='text-align: center; color: black;'>We have analyzed word clouds based sub-Reddit domains due to specificity of the language used</h5>", unsafe_allow_html=True)
        
  

    image_anx_non_stressed = Image.open('images/anxiety_non_stressed.png')
    image_anx_stressed = Image.open('images/anxiety_stressed.png')

    image_ab_non_stressed = Image.open('images/abuse_non_stressed.png')
    image_ab_stressed = Image.open('images/abuse_stressed.png')

    image_fin_non_stressed = Image.open('images/financial_non_stressed.png')
    image_fin_stressed = Image.open('images/financial_stressed.png')

    image_ptsd_non_stressed = Image.open('images/ptsd_non_stressed.png')
    image_ptsd_stressed = Image.open('images/ptsd_stressed.png')

    image_social_non_stressed = Image.open('images/social_non_stressed.png')
    image_social_stressed = Image.open('images/social_stressed.png')


    # col1, col2, col3 = st.columns([20,1,20])
    # with col1:
    #     st.markdown('**Anxiety: Non-Stressed**')
    #     st.image(image_anx_non_stressed)
    # with col3:
    #     st.markdown('**Anxiety: Stressed**')
    #     st.image(image_anx_stressed)

    # col4, col5, col6 = st.columns([20,1,20])
    # with col4:
    #     st.markdown('**Abuse: Non-Stressed**')
    #     st.image(image_ab_non_stressed)
    # with col6:
    #     st.markdown('**Abuse: Stressed**')
    #     st.image(image_ab_stressed)

    col7, col8, col9 = st.columns([20,1,20])
    with col7:
        st.markdown('**Financial: Non-Stressed**')
        st.image(image_fin_non_stressed)
    with col9:
        st.markdown('**Financial: Stressed**')
        st.image(image_fin_stressed)

    # col10, col11, col12 = st.columns([20,1,20])
    # with col10:
    #     st.markdown('**PTSD: Non-Stressed**')
    #     st.image(image_ptsd_non_stressed)
    # with col12:
    #     st.markdown('**PTSD: Stressed**')
    #     st.image(image_ptsd_stressed)

    # col13, col14, col15 = st.columns([20,1,20])
    # with col13:
    #     st.markdown('**Social: Non-Stressed**')
    #     st.image(image_social_non_stressed)
    # with col15:
    #     st.markdown('**Social: Stressed**')
    #     st.image(image_social_stressed)