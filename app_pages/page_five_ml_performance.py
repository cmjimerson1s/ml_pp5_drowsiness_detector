import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_five_body():
    
    version = 'v3'

    st.write(
        f"This page provides an easily understood and visual depiction of"
        f" how the data was divided and how well the model performed."
    )

    st.write("### Image Distribution ###")

    st.warning(
        f"The Drowsy dataset was divided into three subsets.\n\n"
        f"Train set (70% of the whole dataset) is the initial data used to 'fit' or train the model.\n\n"
        f"Validation set (10% of the dataset) helps to improve the model performance by fine-tuning the model after each epoch.\n\n"
        f"Test set (20% of the dataset) informs us about the final accuracy of the model after completing the training phase."
        )
    st.write("---")

    label_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(label_distribution, caption='Label Distribution for Train, Validation, and Test Sets')

    st.write("### Model Performance ###")

    st.write("#### Model History ####")
    st.write("##### Accuracy and Val_Accuracy #####")
    st.warning(
        f"Accuracy is a metric that measures the overall correctness of the" 
        f" model's predictions compared to the true labels in the dataset. It" 
        f" is defined as the ratio of correct predictions to the total number of predictions made by the model.\n\n"
        f"Validation accuracy, often denoted as val_accuracy, is a variant of accuracy used during the model"
        f" training process. When training a machine learning model, it is common to split the dataset into" 
        f" training and validation sets. The training set is used to train the model, while the validation"
        f" set is used to evaluate the model's performance during training."

    )
    model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
    st.image(model_acc, caption='Model Traninig Accuracy')
    st.write("##### Loss and Val_Loss #####")
    st.warning(
        f"Loss, also known as the training loss or training error, is a metric that" 
        f" measures how well the model is performing during the training process. It quantifies" 
        f" the difference between the predicted outputs of the model and the actual ground truth labels in the training data.\n\n"
        f"Validation loss, often denoted as val_loss, is a metric used to evaluate the model's performance" 
        f" on a separate validation dataset during the training process. It measures the difference between" 
        f" the model's predictions and the true labels in the validation set."
    )
    model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
    st.image(model_loss, caption='Model Traninig Losses')

    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))

    st.warning(
        f"Confusion Matrices are used as a performance measurment to dispaly a well trained model."
        f" A well trained model will have a high value for True Negative and True Positive."
        f" True Negative/Positive results represent reality."
        f" A matrix displaying high Fale Positive and False Negative would show a poor performing model."
    )
    
    con_mat = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(con_mat, caption="Confusion Matrix", width=500)