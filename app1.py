import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

def app():
    
    st.markdown("<h3 style='text-align: center; color: black;'>Stress is a nigh-universal human experience particularly in the online world</h3>", unsafe_allow_html=True)
    
    
    st.write("""
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







