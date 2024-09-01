# Example: reuse your existing OpenAI setup
from openai import OpenAI
import streamlit as st

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

st.title("Python Debugger")
st.write("Welcome to the Python Debugging App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Start typing here?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process user input with LM Studio and get response
    with st.chat_message("assistant"):
        st.markdown("Spinning the Hamster Wheel...")
        # Point to the local server
        try:
            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {
                        "role": "system",
                        "content": "Always provide the most efficient python code.",
                    },
                    {"role": "user", "content": "Introduce yourself."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            response = completion.choices[0].message.content

            st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
