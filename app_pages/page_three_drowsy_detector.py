import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_three_body():

    st.info(
        f"To fulfil the current business requirements, you can upload pictures of eyes and detect if the image"
        f" is showing drowsiness. "
    )

    st.write(
        f"You can download an image of either drowsy or awake eyes from [here](https://www.kaggle.com/datasets/hazemfahmy/openned-closed-eyes)."
    )

    st.write(
        "---"
    )

    st.write(
        f"**Upload an image of an eye, you may select more than one.**"
    )

    image_buffer = st.file_uploader('', type='jpeg', accept_multiple_files=True)

    if image_buffer is not None:
        df_report = pd.DataFrame([])
        for image in image_buffer:

            img_pil = (Image.open(image))
            st.info(f"Eye Image to Analyse: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v3'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = pd.concat([df_report,
                                    pd.DataFrame({"Name": [image.name], 
                                                'Result': [pred_class]})],
                                        ignore_index=True)
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
