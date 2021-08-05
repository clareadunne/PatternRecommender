import streamlit as st
from PIL import Image
from Functions import *

st.title("Knitting Pattern Recommender")
#st.write(" # write a thing. ")
#st.write("another thing")

#st.sidebar.selectbox("Select a thing", ('thing 1', 'thing 2'))
#st.sidebar.selectbox("Select a thing", ('thing 3', 'thing 4'))

#is_a_user = 
#not_a_user = st.button('Not a ravelry user.')

#st.write(is_a_user)
#st.write(not_a_user)

rav_user = st.radio('Already a Ravelry user?', options = ('Yes', 'No'))

if rav_user == 'Yes': 
    user = st.text_input('Your Ravelry.com username:') 
    if user != '':
        st.write(get_recommendations(user))

    #st.write(get_recommendations(user))

if rav_user == 'No': 
 
    get_recommendations('not a ravelry user')

#image = Image.open('Vizualisations/Capture3.png')
#st.image(image, caption='Sunrise by the mountains')