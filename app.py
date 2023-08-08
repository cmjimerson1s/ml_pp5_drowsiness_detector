import streamlit as st 
from app_pages.multipage import MultiPage

from app_pages.page_one_summary import page_one_body


app = MultiPage(app_name= "Drowsiness Detector")


app.add_page("Project Summary", page_one_body)


app.run()