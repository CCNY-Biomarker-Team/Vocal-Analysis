
import scipy
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def app():
    st.title("Instructions on Recording Vocal Samples for Vocal Biomarker Tasks")
    #st.write just writes it
    st.write('**Naming Recordings**')
    st.write('Recordings should be titled as: MM-DD-YEAR-#AM/PM-F/M-SJ#-T#')
    st.write('After the date we would like you patients to mention whether the recording was done in the morning hours (AM) or in the afternoon/evening hours (PM).')
    st.write('Please indicate the hour of the day in which the recording task was done like 8PM, 10AM, 11PM')
    st.write('Please indicate your sex along with your subject number (SJ#).')
    st.write('The vocal task for which the recording was done for should be indicated T# (T1 would be for Task 1, the /a/ vowel task and T2 would be Task 2, the passage reading task)')
    st.write('If you have any questions or if you need help, you can ask the researchers on call')

  
