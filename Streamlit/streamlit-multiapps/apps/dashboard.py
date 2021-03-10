
import scipy
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from data.create_data import load_data

def app():
    st.title("SENIOR DESIGN APP DEMO TRIAL RUN :)")
    #st.write just writes it
    st.write('Streamlit is **really cool** :+1: :+1: :+1: :sunglasses:')

    # DISPLAYING DATA 
    data1 = load_data(100000)
    hour= st.slider('hour', 0, 23, 10) #this allows users to change the time instead of setting it at just one time
    data1 = data[data[DATE_TIME].dt.hour == hour] #this allows us to look at pick ups at noon/12 only so it's like a filter

    'Data', data1 #this allows data to be displayed 

    if st.checkbox('Show Raw Data'): # this allows us to hide the raw data table, unless we/users manually click to view it 
        '## Raw Data at %sh' % hour, data1

  
