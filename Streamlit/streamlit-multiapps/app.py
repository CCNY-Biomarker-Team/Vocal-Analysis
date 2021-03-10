import streamlit as st
from multiapp import MultiApp
from apps import home, data_stats, dashboard, data_analysis # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home Main Page", home.app)
app.add_app("Data Stats Page", data_stats.app)
app.add_app("Dashboard Page", dashboard.app)
app.add_app("Data Analysis Page", data_analysis.app)


# The main app
app.run()