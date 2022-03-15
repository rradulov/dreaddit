import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

import app0
import app1
import app2
import app3
import app4
import app5
import app6
import app7
import app8

st.set_page_config(page_title="Dreaddit",layout="wide")

import streamlit as st
PAGES = {
    "Introduction": app0,
    "Project Overview": app1,
    "NLP" : app8,
    "Stress Language": app2,
    "Model Features and Perfomance": app3,
    "Non-Stressed Reddits": app5,
    "Stressed Reddits": app6,
    "Model Insights": app4,
    "Outro":app7
}

st.sidebar.title('Dreaddit')
selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app()


