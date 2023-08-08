import streamlit as st
import matplotlib.pyplot as plt



def page_four_body():
    st.write("### Project Hypothesis and Conclusion")

    st.info(
        f"Hypothesis 1:\n\n"
        f"Drowsy individuals tend to have more closed eyes compared to alert individuals.\n\n"
    )
    st.success(
        f"Conclusion:\n\n"
        f"To validate this hypothesis, we analyze the average eye openness of drowsy and alert" 
        f"individuals from the dataset. Using Machine Learning we are able to determine that the" 
        f"average openness of drowsy eyes is significantly lower than alert eyes. We can see this"
        f"through the Average Image and Variability Images."
    )
    st.info(
        f"Hypothesis 2:\n\n"
        f"There is a visual pattern that can be learned by a machine learning model to classify drowsy and alert eyes with an accuracy of at least 90%."
    )
    st.success(
        f"Conclusion:\n\n"
        f"To validate this hypothesis, we trained a machine learning model on the dataset and evaluated its performance."
        f"The model achieved an accuracy greater than 90% for the test set, and as such, the hypothesis is supported."
    )
