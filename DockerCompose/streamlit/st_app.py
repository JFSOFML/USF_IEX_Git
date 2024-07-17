import streamlit as st 
import pandas as pd
import requests

st.title("Docker Compose Pipeline Demo")

Query = st.text_area(label="Input Text Here") # revisit : Barebones

if st.button("Results"):
    response = requests.post("http://fl_container:5000/query", json={"query":Query}) # Check names if bugs 

    if response.status_code == 200:
        try: 
            fromflask = response.json()
            querysdata = fromflask.get("data", []) # Revisit from flask.py 
            queryscolumns = fromflask.get("columns", [])
            df = pd.DataFrame(querysdata, columns=queryscolumns)
            st.dataframe(df)
        except requests.exceptions.JSONDecodeError:
            st.error("Error: The response is not in JSON format.")
            st.write("Response content:", response.text)
    else:
        st.error(f"Error: Received status code {response.status_code}")
        st.write("Response content:", response.text)