import streamlit as st

import numpy as np
import pandas as pd

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

txt = st.text_area('Text to analyze', '''
     It was the best of times, it was the worst of times, it was
     the age of wisdom, it was the age of foolishness, it was
     the epoch of belief, it was the epoch of incredulity, it
     was the season of Light, it was the season of Darkness, it
     was the spring of hope, it was the winter of despair, (...)
     ''')
st.write('Stressfactor:', run_sentiment_analysis(txt))

txt2 = st.text_area('Text to analyze', '''
     It was the best of times, it was the worst of times, it was
     the age of wisdom, it was the age of foolishness, it was
     the epoch of belief, it was the epoch of incredulity, it
     was the season of Light, it was the season of Darkness, it
     was the spring of hope, it was the winter of despair, (...)
     ''')

txt3 = st.text_area('Text to analyze', '''
     It was the best of times, it was the worst of times, it was
     the age of wisdom, it was the age of foolishness, it was
     the epoch of belief, it was the epoch of incredulity, it
     was the season of Light, it was the season of Darkness, it
     was the spring of hope, it was the winter of despair, (...)
     ''')

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
line_count = st.slider('Select a line count', 1, 10, 3)

# and used in order to select the displayed lines
head_df = df.head(line_count)

head_df
