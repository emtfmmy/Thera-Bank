import pickle

def predict_loan(df):
    pipe = pickle.load(open('final_pipeline.pkl', 'rb'))
    predicted_loan = pipe.predict(df.loc[:,'Age':'CreditCard'])
    df['PersonalLoan'] = predicted_loan
    df['color'] = df['PersonalLoan'].apply(lambda x: 'orange' if x == 1 else 'skyblue')
    return df