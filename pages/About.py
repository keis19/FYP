import streamlit as st

st.title("Welcome To Something's Fishy 🐠")


st.subheader("Information about the project")
st.write("In this current era, payments by credit card have become easier than ever. Although a multitude of improvements have been security has still"
+ "major concern that has transenced with all the major upgrades. With the transactions by credit card and circulations of credit card growing rapidly" 
+ "it calls upon for a better look into security. Our application aims to aid in this process by using top notch Machine Learning algorithms in helping" 
+ "you detect the amount of fraud cases present within your dataset.")

st.subheader("What can you find this application?")
st.write (" Home page - Grey[Here you can upload your fie to get the number of fradulent cases]"
         + "EDA - Grey[Here you can find some interesting data visualization related to your dataset"
         + "Documentation - This page (The One You're In) is where you get details on how to use the application")

st.subheader("How do you use this application")
st.write( "1. Firstly upload your file by clicking Grey[Import file]. red[IMPORTANT NOTE THE FILE SHOULD BE IN CSV FORMAT] "
         +"2. Once the file has been uploaded you wait for a few minutes. Once it's ready to the total number of fradulent cases appears"
         +"3. You can then head to the Exploratory Data Analysis (EDA) section to view some fantastic insights obtained from the dataset.")
         
st.subheader("Want to know the powerful Machine Learning Algorithm (ML) powering this application?")
st.write("The algorithm that we are using to classify fraud cases is the Random Forest Algorithm (RFA). We had conducted a series of tests on the algorithm and sampling."
         + "The best performance with datasets that have undergone Synthetics Minority Over-Sampling Technique (SMOTE) were on K-Neighbors Classifier. Thus, we have used"
         + "that for this application."
         +" Want to learn more about K Neighbors Classifier click this [link] (https://www.ibm.com/my-en/topics/knn)")
         
         
         

         
