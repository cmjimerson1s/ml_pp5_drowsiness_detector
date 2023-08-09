import streamlit as st


def page_one_body():

    st.write("### Project Summary ###")

    st.success(
        "##### Project General Information\n\n"
        "Drowsiness is a significant concern for driver safety as it impairs a driver's ability to stay alert, attentive, and make quick decisions while on the road. It often occurs during long journeys or monotonous driving conditions, such as driving at night or on straight highways. Detecting drowsiness in drivers is crucial to prevent accidents and save lives, as drowsy driving can lead to devastating consequences.\n\n"
    )
    st.info(
        "##### Project Dataset #####\n\n"
        "The dataset used in this project consists of 9,869 JPEG images of 3D rendered eyes, and surrounding orbital features. The total images are divided in half, with half being 'Awake' images and the other half being 'Drowsy'. The background for the images are nuetral gray.\n\n"
    )
    st.warning(
        "##### Business Requirements #####\n\n"
        "This project has three Business Requirements, *a study to visually identify drowsy drivers, accurate prediction of driver drowsiness, downloadable drowsiness prediction report.* Successfully fulfilling these will provide proof of concept and thus allow for further development, with the end goal of providing a real-time video machine learning system to detect drowsiness. Below the Business Requirements are outlined further.\n\n"
        "**1.) A Study to Visually Identify Drowsy Drivers:**\n\n" 
        "The system will perform image analysis on images to visually differentiate between alert and drowsy drivers. It will study the 'openness' of drivers' eyes to identify signs of drowsiness.\n\n"
        "**2.) Accurate Prediction of Driver Drowsiness:**\n\n" 
        "The drowsiness detection model will be developed as a binary classifier to accurately predict whether a given driver is drowsy or not based on their eye images. The model will utilize machine learning algorithms and computer vision techniques to achieve high accuracy in identifying drowsiness in drivers.\n\n"
        "**3.) Downloadable Drowsiness Prediction Reports:**\n\n" 
        "Upon analyzing a driver's eye images, the system will generate a comprehensive prediction report for each examination. The report will include details such as the date and time of the examination, the prediction result (Drowsy or Awake), and the associated probability. The report will be downloadable in a user-friendly format for record-keeping and further analysis."
    )



