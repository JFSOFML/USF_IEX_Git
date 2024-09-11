"""Streamlit display for DockerCompose"""

import time  # Allows adding delays in execution
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
import streamlit as st
import pandas as pd

# Set the title of the app
st.title("Docker Compose Pipeline Demo")

# Prefill the query input box with the SQL query "SELECT * FROM titanic"
query = st.text_area(label="Input Text Here", value="SELECT * FROM titanic")

# Slider for maximum results
max_results = st.sidebar.slider("Max Results", min_value=1, max_value=100, value=10)

# Button to execute the query
if st.button("Results"):
    RESPONSE = None
    # Retry the request up to 5 times if it fails
    for _ in range(5):
        try:
            # Send a POST request to the Flask service with the SQL query
            RESPONSE = requests.post(
                "http://fl_container:5000/query",
                json={"query": f"{query} LIMIT {max_results}"},
                timeout=15,
            )
            if RESPONSE.status_code == 200:
                st.success("Query executed successfully!")
                st.balloons()
                break
        except RequestsConnectionError:
            st.warning("Waiting for the Flask service to be available...")
            time.sleep(5)  # Wait for 5 seconds before retrying

    # Process the response from the Flask service
    if RESPONSE and RESPONSE.status_code == 200:
        try:
            # Parse the JSON response
            from_flask = RESPONSE.json()
            query_data = from_flask.get("data", [])
            query_columns = from_flask.get("columns", [])

            #
            # Create a DataFrame from the response data
            df = pd.DataFrame(query_data, columns=query_columns)

            # Display the DataFrame in the app
            st.dataframe(df)

        except requests.exceptions.JSONDecodeError:
            st.error("Error: The response is not in JSON format.")
            st.write("Response content:", RESPONSE.text)
    else:
        if RESPONSE:
            st.error(f"Error: Received status code {RESPONSE.status_code}")
            st.write("Response content:", RESPONSE.text)
        else:
            st.error("Error: Could not connect to the Flask service")