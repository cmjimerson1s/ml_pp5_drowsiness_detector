import streamlit as st 
from app_pages.multipage import MultiPage

from app_pages.page_one_summary import page_one_body
from app_pages.page_two_visualizer import page_two_body
from app_pages.page_three_drowsy_detector import page_three_body
from app_pages.page_four_hypothesis import page_four_body



app = MultiPage(app_name= "Drowsiness Detector")


app.add_page("Project Summary", page_one_body)
app.add_page("Drowsiness Visualization", page_two_body)
app.add_page("Drowsiness Detector", page_three_body)
app.add_page("Project Hypothesis", page_four_body)


app.run()