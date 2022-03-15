import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():

    st.markdown("<h6 style='text-align: center; padding:0px; color: black;'>Most impactful model features examples</h6>", unsafe_allow_html=True)
    
    image_vars = Image.open('images/var_boxplots1.png')
    st.image(image_vars, caption='', use_column_width=True)

    st.write("###")
      
    st.markdown("<h6 style='text-align: center; padding:0px; color: black;'>Stacking Classfier Ensemble Model Confusion Matrix</h6>", unsafe_allow_html=True)
    image_conf_matrix = Image.open('images/conf_matrix.png')
    
    st.write("###")
    
    col1, mid, col2 = st.columns([10,1,20])
    with col1:
        st.image(image_conf_matrix)
    with col2:
         st.markdown("<h6 style='text-align: justify ; padding:0px; color: grey;'>Our best performing model is able to correctly classify stressed people more accurate than non-stressed ones</h6>", unsafe_allow_html=True)
         st.write("###")
         st.markdown("<h6 style='text-align: justify ; padding:0px; color: grey;'>Our Best Performing Stacking Classifier is a combination of SVC, Gradient Boosting and Logistic regression </h6>", unsafe_allow_html=True)
         st.write("###")
         st.markdown("<h6 style='text-align: justify ; padding:0px; color: grey;'>Best accuracy <b>76.5%</b></h6>", unsafe_allow_html=True)
         
        # st.markdown("""The diagonal elements represent the proportion of observations where
        #          the predicted label is equal to the true label, while off-diagonal elements 
        #          are those that are mislabeled by the classifier.
        #          Our best performing model is able to correctly classify stressed people more accurate than
        #          non-stressed ones.""")