import os
import streamlit as st
import subprocess
import webbrowser
from dotenv import load_dotenv
load_dotenv(".env.groq", override=True)
# Set up the Streamlit app
st.title("AI Clinic Agents")

# Create buttons
if st.button("General Inquiry Agent"):
    subprocess.Popen(["streamlit", "run", "GeneralInquiryAgent.py"])
    webbrowser.open_new("http://localhost:8502")  # Adjust port if needed
if st.button("Info Agent"):
    subprocess.Popen(["streamlit", "run", "InfoAgent.py"])
    webbrowser.open_new("http://localhost:8503")  # Adjust port if needed