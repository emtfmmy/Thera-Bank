import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from model import predict_loan
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split  
#################################################################
################## Page Settings ################################
#################################################################
st.set_page_config(page_title="My Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

#################################################################
################## Page Header ##################################
#################################################################
st.header("Predicting Potential Customers to Make Loan")
st.write("Our application uses Artificial Intelligence to predict potential customer to purchase the loan.")
st.markdown('---')

################## Sidebar Menu #################################
page_selected = st.sidebar.radio("Menu", ["Home", "Model"])

################################################################
################## Home Page ###################################
################################################################
if page_selected == "Home":
    
    ######### Load labeled data from datastore #################
    df = predict_loan(pd.read_csv('Bank_Personal_Loan_Test1.csv'))
    
    ############# Filters ######################################
    
    ######### Date range slider ################################
    start, end = st.sidebar.select_slider(
                    "Select Age Range", 
                    df.Age.drop_duplicates().sort_values(), 
                    value=(df.Age.min(), df.Age.max()))

    ######### Apply filters ####################################
    df_filter = df.loc[(df.Age >= start) & (df.Age <= end), :]
    
    if df_filter.shape[0] > 0:
        ######### Main Story Plot ###################################
        col1, col2 = st.columns((2,1))
        with col1: 
            #print(df_filter['CreditCard'])
            st.bar_chart(df_filter['PersonalLoan'].value_counts())
        with col2:
            st.write("This plot shows the prediction number of customers who accept the loan and customers who don't accept the loan.")
        st.markdown('---')
        
        ######### Income vs CreditCard Plot ###################################
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = df_filter.plot.scatter(x='Age', y='Income',  c=df_filter['color'], figsize=(6,2))
            st.pyplot(ax.figure)
        with col2:
            st.write("This scatter plot shows the relationship between Income and Age among people. Orange represents people who have loan, while blue represents people who don't have loan. It shows people who have loan are always in high income group.")
        st.markdown('---')

        ######### Education vs Loan Plot ###################################
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(df_filter.Education, df_filter.PersonalLoan).plot(
                    kind="bar", 
                    figsize=(6,2), 
                    xlabel = "Education",
                    color={0:'skyblue', 1: 'orange'})
            st.pyplot(ax.figure)
        with col2:
            st.write('This bar plot shows the number of customers who buy a loan in different level of education.')
        st.markdown('---')

        ######### Education vs Loan Plot(2) ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        df_a = df_filter.groupby(by = 'Family')[['PersonalLoan']].sum()
        df_a1 = df_filter.groupby(by = 'Family')[['PersonalLoan']].count()
        df_a['percent'] = df_a.PersonalLoan/df_a1.PersonalLoan
        df_a['Family'] = df_a.index
        ax = df_a.plot.bar(x="Family",
        y="percent",
        figsize=(6,2), 
        xlabel = "Family",
        color = "green")
        st.pyplot(ax.figure)
    with col2:
        st.write('This bar plot shows the number of customers who buy a loan in different family size.')
    st.markdown('---')


################################################################
############### Model Process Describe #########################
################################################################
else:
    st.subheader('Training dataset introduce')
    st.write('This case is about a bank (Thera Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). The department wants to build a model that will help them identify the potential customers who have a higher probability of purchasing the loan. The data set includes the following independent and dependent variables')
    st.write('')
    st.write('Independent variable:')
    st.write("Age: Customer's age in completed years")
    st.write("Experience: #years of professional experience")
    st.write("Income: Annual income of the customer ($000)")
    st.write("Family: Family size of the customer")
    st.write("CCAvg: Avg. spending on credit cards per month ($000)")
    st.write("Education: Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional")
    st.write("Mortgage: Value of house mortgage if any. ($000)")
    st.write("Securities Account: Does the customer have a securities account with the bank?")
    st.write("CD Account: Does the customer have a certificate of deposit (CD) account with the bank?")
    st.write("Online: Does the customer use internet banking facilities?")
    st.write("CreditCard: Does the customer use a credit card issued by UniversalBank?")
    st.write('')
    st.write('Dependent Variable:')
    st.write("Personal Loan: Did this customer accept the personal loan offered in the last campaign?")
    st.markdown('---')
    st.subheader('data explore of training dataset')
    df_training = pd.read_csv('Bank_Personal_Loan_Train1.csv')
    col1, col2, col3, col4 = st.columns((1,1,1,1))
    with col1:
        ax1 = df_training[['Age']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax1.figure)
        ax2 = df_training[['Experience']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax2.figure)
        ax3 = df_training[['Income']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax3.figure)
    with col2:
        ax4 = df_training[['Family']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax4.figure)
        ax5 = df_training[['CCAvg']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax5.figure)
        ax6 = df_training[['Education']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax6.figure)
    with col3:
        ax7 = df_training[['Mortgage']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax7.figure)
        ax8 = df_training[['Securities Account']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax8.figure)
        ax9 = df_training[['CD Account']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax8.figure)
    with col4:
        ax10 = df_training[['Online']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax7.figure)
        ax11 = df_training[['CreditCard']].plot.hist(bins=15, figsize=[20,10])
        st.pyplot(ax8.figure)
    st.table(df_training.describe())
    st.markdown('---') 
    st.subheader('Data processing and modeling')
    st.write("From the above images, we could find the range of numeric data is different and some data is skew. So firstly I normalize data. Then I use 'OneHotEncoder' to transform categorical data. Secondly, I try LogisticRegression, KNeighborsClassifier and RandomForestClassifier model. And use accuracy_score to test its accuracy. Finally, pick the one with the highest accuracy up to save as final pipeline. After building model, use the final model to predict test data and get result.")
    st.write('')
    st.subheader('Model Evaluate')
    st.write('I evaluated the model by accuracy rate, recall rate. In order to show these values more intuitively, I show confusion matrix here. Recall rate is important in the prediction, because I hope those real patients are predicted as much as possible ')
    pipe = pickle.load(open('final_pipeline.pkl', 'rb'))
    X_train, X_test, y_train, y_test = train_test_split(df_training.loc[:,'Age':'CreditCard'], df_training['PersonalLoan'], test_size = 0.3, stratify=df_training['PersonalLoan']) 
    train_predict = pipe.predict(X_test)
    cm = confusion_matrix(y_test, train_predict, labels=pipe.classes_)
    col1, col2 = st.columns((1,1))
    with col1:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['without loan','with loan']) 
        ax9 = disp.plot()
        st.pyplot(ax9.figure_)
    with col2:
        st.write('')
    st.write('accuracy:'+ ' '+ str(accuracy_score(y_test,train_predict).round(4)*100)+'%')
    st.write('recall:'+ ' '+ str((cm[1,1]/(cm[1,0] + cm[1,1])).round(4)*100)+'%')
    st.write('Both accuracy and recall rate are above 90%, so the performance of model performs meets the expectation')
    

