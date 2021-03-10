import pandas as pd
import numpy as np
import streamlit as st


def create_table(n=7):
    df = pd.DataFrame({"x": range(1, 11), "y": n})
    df['x*y'] = df.x * df.y
    return df


# LOADING SAMPLE DATA, (for cognative fatigue tasks if we finsih the vocal part of the app)
DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz")

@st.cache(persist=True) #code like this is usually slow to run b/c have to go web and download and transform etc. 
#but it's not now, b/c of this st.cache function that stores the data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME]) #convert one column of string data to date/time
    return data
