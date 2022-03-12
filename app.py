import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

st.set_page_config(page_title="Dreaddit",layout="wide")


st.title('Dreaddit Stress Analysis')
st.header("""Stressed? Find it out by Social Media texts!""")
st.subheader("""
Stress is a nigh-universal human experience particularly in the online world.
While stress can be a motivator, too much stress is associated with many
negative health outcomes, making its identification useful across a range
of domains. However, existing computational research typically only studies
stress in domains such as speech, or in short genres such as Twitter.""")

st.write("""
We present Dreaddit, a new text corpus of lengthy multi-domain social media
data for the identification of stress. Our dataset consists of 190K posts
from five different categories of Reddit communities; we additionally label
3.5K total segments taken from 3K posts using Amazon Mechanical Turk.
We present preliminary supervised learning methods for identifying stress,
both neural and traditional, and analyze the complexity and diversity of the
data and characteristics of each category""")


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


col1, col2, col3 = st.columns([20,1,20])
with col1:
    st.subheader('Anxiety: Non-Stressed')
    st.image(image_anx_non_stressed)
with col3:
    st.subheader('Anxiety: Stressed')
    st.image(image_anx_stressed)

col4, col5, col6 = st.columns([20,1,20])
with col4:
    st.subheader('Abuse: Non-Stressed')
    st.image(image_ab_non_stressed)
with col6:
    st.subheader('Abuse: Stressed')
    st.image(image_ab_stressed)

col7, col8, col9 = st.columns([20,1,20])
with col7:
    st.subheader('Financial: Non-Stressed')
    st.image(image_fin_non_stressed)
with col9:
    st.subheader('Financial: Stressed')
    st.image(image_fin_stressed)

col10, col11, col12 = st.columns([20,1,20])
with col10:
    st.subheader('PTSD: Non-Stressed')
    st.image(image_ptsd_non_stressed)
with col12:
    st.subheader('PTSD: Stressed')
    st.image(image_ptsd_stressed)

col13, col14, col15 = st.columns([20,1,20])
with col13:
    st.subheader('Social: Non-Stressed')
    st.image(image_social_non_stressed)
with col15:
    st.subheader('Social: Stressed')
    st.image(image_social_stressed)


image_vars = Image.open('images/var_boxplots1.png')
st.image(image_vars, caption='Features distributions', use_column_width=True)



image_conf_matrix = Image.open('images/conf_matrix.png')
st.image(image_conf_matrix, caption='Confusion Matrix', use_column_width=False)

from dreaddit.predict import Predictor
predict_class = Predictor(model_path='dreaddit/model.joblib');

st.latex(r'''
     residual = | predict\_proba - y\_true |
     ''')

predict_class.cleaned_output_df.sort_values(by=['confidence','residual'],ascending=False).head(3)[['text','residual']]

predict_class.cleaned_output_df.sort_values(by=['residual'],ascending=True).head(3)[['text','residual']]










