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

st.set_page_config(page_title="Dreaddit",layout="wide")

import streamlit as st
PAGES = {
    "Introduction": app0,
    "Project Overview": app1,
    "Stress Language": app2,
    "Model Features and Perfomance": app3,
    "True Negatives": app5,
    "True Positives": app6,
    "Model Insights": app4
    
    
    
}
st.sidebar.title('Dreaddit')
selection = st.sidebar.radio("", list(PAGES.keys()))
page = PAGES[selection]
page.app()


