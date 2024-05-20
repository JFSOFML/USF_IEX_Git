import streamlit as st
import pickle

st.title("Pytorch!")
st.divider()
st.write("""
### Model Loss & Accuracy
""")
st.write("""
Test Loss: 0.0012

Accuracy: 98.03%

""")



st.divider()

st.header("Training Loop")
st.write("""
Epoch 0: Loss: 0.1448
        
Epoch 1: Loss: 0.0311
        
Epoch 2: Loss: 0.0391
        
Epoch 3: Loss: 0.0193
        
Epoch 4: Loss: 0.0075
        
Epoch 5: Loss: 0.0761
        
Epoch 6: Loss: 0.0200
        
Epoch 7: Loss: 0.0181
        
Epoch 8: Loss: 0.0001
        
Epoch 9: Loss: 0.0005

        """)

st.divider()

st.header("Visulizations")

st.subheader("50th Index of dataset")
st.image("Pictures\MNIST_img.png")

st.balloons()
