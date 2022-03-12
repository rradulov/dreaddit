import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from itertools import cycle

st.set_page_config(page_title="Dreaddit",layout="wide")

# def paginator(label, items, items_per_page=10, on_sidebar=True):
#     """Lets the user paginate a set of items.
#     Parameters
#     ----------
#     label : str
#         The label to display over the pagination widget.
#     items : Iterator[Any]
#         The items to display in the paginator.
#     items_per_page: int
#         The number of items to display per page.
#     on_sidebar: bool
#         Whether to display the paginator widget on the sidebar.
        
#     Returns
#     -------
#     Iterator[Tuple[int, Any]]
#         An iterator over *only the items on that page*, including
#         the item's index.
#     Example
#     -------
#     This shows how to display a few pages of fruit.
#     >>> fruit_list = [
#     ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
#     ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
#     ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
#     ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
#     ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
#     ... ]
#     ...
#     ... for i, fruit in paginator("Select a fruit page", fruit_list):
#     ...     st.write('%s. **%s**' % (i, fruit))
#     """

#     # Figure out where to display the paginator
#     if on_sidebar:
#         location = st.sidebar.empty()
#     else:
#         location = st.empty()

#     # Display a pagination selectbox in the specified location.
#     items = list(items)
#     n_pages = len(items)
#     n_pages = (len(items) - 1) // items_per_page + 1
#     page_format_func = lambda i: "Page %s" % i
#     page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

#     # Iterate over the items in the page to let the user display them.
#     min_index = page_number * items_per_page
#     max_index = min_index + items_per_page
#     import itertools
#     return itertools.islice(enumerate(items), min_index, max_index)

# def demonstrate_paginator():
#     fruit_list = [
#         'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
#         'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
#         'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
#         'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
#         'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
#     ]
#     for i, fruit in paginator("Select a fruit page", fruit_list):
#         st.write('%s. **%s**' % (i, fruit))
# filteredImages = [image_anx,image_ab,image_fin, image_ptsd, image_social] 
# caption = ['Anxiety', 'Abuse', 'Financial', 'PTSD', 'Social'] 
# cols = cycle(st.columns(5)) 
# for idx, filteredImage in enumerate(filteredImages):
#     next(cols).image(filteredImage, width=150, caption=caption[idx])

# word_clouds = [image_anx,image_ab,image_fin, image_ptsd, image_social]
# caption=['Anxiety', 'Abuse', 'Financial', 'PTSD', 'Social']

# image_iterator = paginator("Select page", word_clouds)
# indices_on_page, images_on_page = map(list, zip(*image_iterator))
# st.image(images_on_page, width=200, caption=caption) #indices_on_page)


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

st.latex(r'''
     residual = | predict\_proba - y\_true |
     ''')

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


st.write(predict_class.cleaned_output_df.sort_values(by=['confidence','residual'],ascending=False).head(3))

st.text(predict_class.cleaned_output_df.loc[322].text)








# name = st.text_input("Name of Passenger ")
# sex = st.selectbox("Sex",options=['Male' , 'Female'])
# age = st.slider("Age", 1, 100,1)
# p_class = st.selectbox("Passenger Class",options=['First Class' , 'Second Class' , 'Third Class'])

# sex = 0 if sex == 'Male' else 1
# f_class , s_class , t_class = 0,0,0
# if p_class == 'First Class':
# 	f_class = 1
# elif p_class == 'Second Class':
# 	s_class = 1
# else:
# 	t_class = 1
# input_data = scaler.transform([[sex , age, f_class , s_class, t_class]])
# prediction = model.predict(input_data)
# predict_probability = model.predict_proba(input_data)

# if prediction[0] == 1:
# 	st.subheader('Passenger {} would have survived with a probability of {}%'.format(name , round(predict_probability[0][1]*100 , 3)))
# else:
# 	st.subheader('Passenger {} would not have survived with a probability of {}%'.format(name, round(predict_probability[0][0]*100 , 3)))



# st.write('You selected:', text)
