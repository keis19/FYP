import streamlit as st

st.title("Welcome To Something's Fishy 🐠")


st.subheader("Information about the project")
st.write("Credit card fraud detection is a pressing issue in the financial industry. With the increasing use of credit cards and online transactions, the opportunity for fraudulent activities also increases. Fraudulent activities in credit card transactions not only result in significant financial losses for individuals and businesses, but also undermine the trust in the financial system. The use of stolen or fake credit card information, phishing scams and skimming are just some examples of credit card fraud. It is crucial to detect and prevent these fraudulent activities in order to protect consumers and businesses from financial loss and to maintain trust in the financial system."
+ "The importance of tackling credit card fraud cannot be overstated. Financial institutions and businesses bear the direct financial loss from fraudulent activities, but consumers also suffer from the consequences. For example, their credit score may be negatively affected or they may have to go through the inconvenience of having their credit card cancelled and reissued. Furthermore, credit card fraud can also have a ripple effect on the economy by eroding trust in the financial system, which in turn can lead to reduced investment and spending."
+ "As the number and sophistication of fraudulent activities continue to increase, traditional methods of detection are becoming less effective. This is where machine learning models come in as a solution. The machine learning algorithm that been instilled in this application is the KNN Classifier Algorithm.")

st.subheader("What can you find this application?")
st.write ("Home page -Here you can upload your fie to get the number of fradulent cases")
st.write("EDA - Here you can find some interesting data visualization related to your dataset")
st.write("About - This page (The One You're In) is where you get details on how to use the application")

st.subheader("How do you use this application")
st.write( "1. Firstly upload your file by clicking Import file. (IMPORTANT NOTE THE FILE SHOULD BE IN CSV FORMAT) ")
st.write( "2. Once the file has been uploaded you wait for a few minutes. Once it's ready to the total number of fradulent cases appears")
st.write("3. You can then head to the Exploratory Data Analysis (EDA) section to view some fantastic insights obtained from the dataset.")
         
st.subheader("Want to know the powerful Machine Learning Algorithm (ML) powering this application?")
st.write("The algorithm that we are using to classify fraud cases is the K Neighbors Classifier. We had conducted a series of tests on the algorithm and sampling."
         + "The best performance with datasets that have undergone Synthetics Minority Over-Sampling Technique (SMOTE) were on K-Neighbors Classifier. Thus, we have used"
         + " that for this application."
         +" Want to learn more about K Neighbors Classifier click this [link](https://www.ibm.com/my-en/topics/knn)")
         
         
         

         
