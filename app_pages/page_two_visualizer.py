import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def page_two_body():
    st.write("### Drowsiness Visualizer")
    st.info(
        f"To fulfill the first business requirement we have created visualization to differentiate between an eye that" 
        f" is showing signs of fatigue or 'drowsy', and an eye that is alert and 'awake'."
    )

    version = 'v3'
    if st.checkbox("Difference between average and variability image"):

        avg_awake = plt.imread(f"outputs/{version}/avg_var_Awake.png")
        avg_drowsy = plt.imread(f"outputs/{version}/avg_var_Drowsy.png")

        st.warning(
            f"*We notice the average and variability images didn't show "
            f"patterns where we could intuitively differentiate one to another."
            f" However, awake images do apear to show a higher amount of white")
        
        st.image(avg_awake, caption="Awake: Average and Variability")
        st.image(avg_drowsy, caption="Drowsy: Average and Variability")
        st.write("---")
    
    if st.checkbox("Differences between average 'Awake' image, and average 'Drowsy' image."):

        diff_between_avgs = plt.imread(f'outputs/{version}/avg_diff.png')

        st.warning(
            f"We notice this study didn't show patters where we could intuitavely"
            f" differentiate between Awake and Drowsy")
        st.image(diff_between_avgs, caption="Difference between average images for Awake and Drowsey")

    if st.checkbox("Image Montage"):
        my_data_dir = 'inputs/drowsiness'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select either Drowsy, or Awake", options=labels, index=0)
        if st.button("View Montage"):
            montage = plt.imread(f'outputs/{label_to_display}Montage.png')
            st.image(montage)
        st.write("---")


