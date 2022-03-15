import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():
    
    st.markdown("<h3 style='text-align: left; color: black;'>Stress is a universal human experience particularly in the online world</h3>", unsafe_allow_html=True)
    
    
    st.write("###")
    
    st.markdown("<h5 style='text-align: justify; color: black;'>Too much stress is associated with many negative health outcomes, therefore making its identification useful across a range of domains.</h5>", unsafe_allow_html=True)
    
    st.write("###")
    
    st.markdown("<h5 style='text-align: justify; color: black;'>Identifying stressed people allows them to be targeted with helpful resources or de-stressing content. A staggering 70 million work days are lost each year due to mental health problems in the UK, costing employers approximately £2.4 billion per year. </h5>", unsafe_allow_html=True)
    
    st.write("###")
    
    st.markdown("<h5 style='text-align: justify; color: black;'>We present Dreaddit - a text corpus of lengthy multi-domain social media (Reddit) data for the identification of stress. The data set consists of 3.5K labelled segments by human annotators.</h5>", unsafe_allow_html=True)
    
    st.write("###")
    
    # st.markdown("<h5 style='text-align: justify; color: black;'>We have applied machine learning and deep learning models in order to predict whether a person is stressed or not based on their Reddit post as well as analyze the complexity and diversity of languge.</h5>", unsafe_allow_html=True)
    
    # st.write("###")

    st.markdown("<h5 style='text-align: justify; color: black;'> Machine Learning techniques with contect specific vectorizer outperform our Deep Learning models. Our best performing stacking classfier ensamble with built-in automated feature selection process has a combination of lexical and social media features extracted from the Reddit posts.</h5>", unsafe_allow_html=True)




