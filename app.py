import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Quick reference", # => Quick reference - Streamlit
    page_icon="🐍",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed

CSS = """
title {
    color: red;
}
.stApp {
    background-color: white;
}
"""

if st.checkbox('Inject CSS'):
    st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.title('Dreaddit Stress Analysis')
st.markdown("""# Stressed? Find it out by Social Media texts!""")
st.text("""
Stress is a nigh-universal human experience particularly in the online world.
While stress can be a motivator, too much stress is associated with many
negative health outcomes, making its identification useful across a range
of domains. However, existing computational research typically only studies
stress in domains such as speech, or in short genres such as Twitter.
We present Dreaddit, a new text corpus of lengthy multi-domain social media
data for the identification of stress. Our dataset consists of 190K posts
from five different categories of Reddit communities; we additionally label
3.5K total segments taken from 3K posts using Amazon Mechanical Turk.
We present preliminary supervised learning methods for identifying stress,
both neural and traditional, and analyze the complexity and diversity of the
data and characteristics of each category""")
placeholder = st.empty()

name = st.text_input("Name of Passenger ")
sex = st.selectbox("Sex",options=['Male' , 'Female'])
age = st.slider("Age", 1, 100,1)
p_class = st.selectbox("Passenger Class",options=['First Class' , 'Second Class' , 'Third Class'])

sex = 0 if sex == 'Male' else 1
f_class , s_class , t_class = 0,0,0
if p_class == 'First Class':
	f_class = 1
elif p_class == 'Second Class':
	s_class = 1
else:
	t_class = 1
input_data = scaler.transform([[sex , age, f_class , s_class, t_class]])
prediction = model.predict(input_data)
predict_probability = model.predict_proba(input_data)

if prediction[0] == 1:
	st.subheader('Passenger {} would have survived with a probability of {}%'.format(name , round(predict_probability[0][1]*100 , 3)))
else:
	st.subheader('Passenger {} would not have survived with a probability of {}%'.format(name, round(predict_probability[0][0]*100 , 3)))



st.write('You selected:', text)
