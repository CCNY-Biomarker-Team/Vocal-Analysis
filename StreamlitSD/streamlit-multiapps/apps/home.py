import streamlit as st
import pandas as pd
import numpy as np
from data.create_data import create_tables

def app():
    #st.title('Home')
    #st.write("This is the home page")
    
    st.markdown("### Sample Data")
    df = create_tables()
    st.write(df)

    st.write('Navigate to `instruction` page to start recording!')




