import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)
def predict_note_authentication(Age,DA,Wallet,AvgDep):
     prediction=classifier.predict([[(Age,DA,Wallet,AvgDep)]])
    #prediction = np.array([[(prediction)]])
   # output = round(prediction[0],2)
     print(prediction)
     return prediction

def main():
    st.title("Fraud Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Fraud Detection ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.text_input("Age")
    #Level = st.text_input("Level","Type Here")
   # Sports_Event = st.text_input("Sports_Event","Type Here")
   # League = st.text_input("League","Type Here")
    
   # Deposit = st.text_input("Deposit","Type Here")
   # First_Dep_Date = st.text_input("First_Dep_Date","Type Here")
   # Mode_of_Payment = st.text_input("Mode_of_Payment","Type Here")
   # Bonus_type = st.text_input("Bonus_type","Type Here")
    
    #Bonus_type = st.text_input("Bonus_type","Type Here")
    DA = st.text_input("DA")
    Wallet = st.text_input("Wallet")
    AvgDep = st.text_input("AvgDep")
    
    #Dep_Amt_Approved = st.text_input("Dep_Amt_Approved","Type Here")
    #Dep_Amt_Rejected = st.text_input("Dep_Amt_Rejected","Type Here")
   # AR = st.text_input("AR","Type Here")
    
   
    output=""
    if st.button("Predict"):
        output=predict_note_authentication(Age,DA,Wallet,AvgDep)
        #output = round(prediction)
        #result=predict_note_authentication(Age,DA,Wallet,AvgDep)
        #result = np.array(result).reshape(1,20)
    st.success('The output is {}'.format(output))
    
if __name__=='__main__':
    main()