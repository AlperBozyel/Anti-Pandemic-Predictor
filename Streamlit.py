# -*- coding: utf-8 -*-


import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def Read_csv():
    dataset = pd.read_csv('Anket_fin.csv')
    df_s = dataset.copy()
    df = pd.DataFrame(df_s)
    return df

def Split_Column(df):
    Ind = ["Age","Sex","Education","Income","GM4","CM4","MSS","RA1","PS2"
           ,"BPS1","BPS2","BPS4","BPS6"]
    df_Ind = df[Ind]
    Target_Par = ["RA3_bin"]
    df_Target = df[Target_Par]
    return df_Ind, df_Target

def Merge_df(df_Ind, data):
    frames = [df_Ind, data]
    df_Ind_F = pd.concat(frames)    
    return df_Ind_F

def PCA_I(df_Ind):
    pca = PCA(n_components = (8))
    X_pca = pca.fit_transform(df_Ind)
    X_pca_df = pd.DataFrame(X_pca)
    data = X_pca_df.tail(n=1)
    X_pca_df = X_pca_df.head(n=117)
    return X_pca_df, data

def Binary_Logistic_Regression(X_pca, df_Target):
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, df_Target, test_size=0.2, random_state=30)
    model_pca = LogisticRegression()
    model_pca.fit(X_train_pca, y_train.values.ravel())
    model_pca.score(X_test_pca, y_test)
    return model_pca

def Result_Output(Model):
    Predict = Model.predict(data)
    if (Predict==1):
        result = "You are not anti-pandemic"
    else:
        result = "You are anti-pandemic"
    return result
    
st.write("""
         
         # Simple Anti-Pandemic Behaviour Prediction App

    This app predicts that are you anti-pandemic side or not  
         
         
     """)

st.sidebar.header('User Input Parameters')
st.sidebar.write(""" Demografik Bilgiler """)

def user_input_features():
    
    st.sidebar.write("""
                     
                     1: Erkek, 2: Kadin
                     
                     """) 
    Cins = st.sidebar.selectbox('Cinsiyet',('1','2'))
    st.sidebar.write("""
                     
                     1: 0-18, 2: 19-24, 3: 25-34, 4: 35-49, 5: 50-64, 6: 65+
                     
                     """)     
    Yas = st.sidebar.selectbox('Yasiniz',("1","2", "3", "4","5","6"))
    st.sidebar.write("""
                     
                     1: lkokul mezunu, 2: Ortaokul mezunu, 3: Lise mezunu, 4: Lisans mezunu, 5: Yuksek Lisans mezunu, 6: Doktora mezunu
                     
                     """)    
    Egitim = st.sidebar.selectbox('Egitiminiz',("1", "2","3","4","5","6")) 
    st.sidebar.write("""
                     
                     0: Ogrenciyim/ gecimimi baskasindan sagliyorum., 1:0 -4250 tl, 2: 4251- 6750 tl, 3: 6751 - 8250 tl,
                     4: 8251 - 10750 tl, 5: 10750 - 15000 tl, 6: 15000tl+
                     
                     """)     
    Gelir = st.sidebar.selectbox('Geliriniz',("0", "1", "2","3", "4", "5", "6"))

    st.sidebar.write("""
                       
                     
                     Gorusler
                     
                     
                     """)

    st.sidebar.write("""
                     1: Katılmıyorum, 2: Kararsızım, 3:Katılıyorum 
                     """)    
    GM4 = st.sidebar.selectbox('Pandemi ile ilgili dogru bilgileri geleneksel medya kanallarindan ogrendim.',('1','2','3'))    
    CM4 = st.sidebar.selectbox('Pandemi ile ilgili dogru bilgileri cevremdeki kisilerden ogrendim.',('1','2','3'))
    st.sidebar.write("""
                     
                     1: Televizyon,Radio gibi Geleneksel kanallardan, 
                     2: Twitter, Facebook gibi Toplumsal kanallardan,
                     3: Aile,Arkadaş gibi Çevresel kanallardan
                     
                     """)     
    MSS = st.sidebar.selectbox('Pandemi ile ilgili en güvenilir bilgiye hangi kanal uzerinden erisebildiginizi dusunuyorsunuz?',('1','2','3'))    
    RA1 = st.sidebar.selectbox('Pandemi boyunca, Yasadıgım bolgede coronavirus salgını cok ciddi seviyedeydi?',('1','2','3'))
    PS2 = st.sidebar.selectbox('Pandemi ile mücadelede aşının etkili oldugunu düsünüyorum.',('1','2','3'))    
    BPS1 = st.sidebar.selectbox('Pandemi boyunca toplum içinde daima maske taktim.',('1','2','3'))   
    BPS2 = st.sidebar.selectbox('Eve geldiginizde daima ellerimi en az 20sn. yıkadim.',('1','2','3'))    
    BPS4 = st.sidebar.selectbox('Pandemi ile ilgili dogru bildigim bilgileri baskalariyla da paylastim.',('1','2','3'))
    BPS6 = st.sidebar.selectbox('Pandemiyi onlemeye yonelik faaliyetlere gonullu katildim.',('1','2','3'))    
    
    button = st.sidebar.button('Predict')

    data = {    'Age' : Yas,
                'Sex' : Cins,
                'Education' : Egitim,
                'Income' : Gelir,
                'GM4': GM4,
                'CM4': CM4,
                'MSS': MSS,
                'RA1': RA1,
                'PS2': PS2,
                'BPS1': BPS1,
                'BPS2': BPS2,
                'BPS4': BPS4,
                'BPS6': BPS6
                }
    
    features = pd.DataFrame(data, index=[0],dtype=('float64'))
    return features, button
    
df,button = user_input_features()

st.subheader('User Input Parameters')
st.write(df)


if (button):
    
    dataset = Read_csv()
    
    df_Ind, df_Target = Split_Column(dataset)
    
    df_Ind_merged = Merge_df(df_Ind, df)
    
    X_pca, data = PCA_I(df_Ind_merged)    
    
    Model = Binary_Logistic_Regression(X_pca, df_Target)
 
    st.subheader('Class Labels')
    st.write("""
            1: You are not anti-pandemic
            
            -1: You are anti-pandemic
        """)

    st.subheader('Prediction')

    result = Result_Output(Model)
    st.write(result)

else:
    st.subheader('Please fill in the expressions on the left')
    st.write("""
           After filling out the expressions, simply press the Predict button.
        """)    