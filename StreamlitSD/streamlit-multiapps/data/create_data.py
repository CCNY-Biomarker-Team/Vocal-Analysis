import pandas as pd
import numpy as np
import streamlit as st


def create_tables(n=7):
    df = pd.DataFrame({"x": range(1, 11), "y": n})
    df['x*y'] = df.x * df.y
    return df

