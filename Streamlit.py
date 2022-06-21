   # -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def Read_csv(Survey_name):
    dataset = pd.read_csv(Survey_name)
    df_s = dataset.copy()
    df = pd.DataFrame(df_s)
    return df

def Encoding(df):
    cleanup_nums = {'Age': {"0-18": 1, "19-24": 2, "25-34": 3, "35-49": 4,"50-64": 5,"65+": 6},
                    'Sex': {"Erkek": 1, "Kadın": 0},                    
                    'Education': {"Ilkokul mezunu": 1, "Ortaokul mezunu": 2,"Lise mezunu": 3,"Lisans mezunu": 4,"Yuksek Lisans mezunu": 5,"Doktora mezunu": 6},
                    'Income': {"Ogrenciyim/ gecimimi baskasindan sagliyorum.": 0, "0 -4250 tl": 1, "4251- 6750 tl": 2,"6751 - 8250 tl": 3, "8251 - 10750 tl": 4, "10750 - 15000 tl": 5, "15000tl+": 6},
                    'GM4': {"Evet": 1, "Hayır": 0},
                    'CM4': {"Evet": 1, "Hayır": 0},
                    'MSS':  {"Geleneksel Kanal": 1, "Toplumsal Kanal": 2, "Cevresel Kanal": 3},
                    'RA1':  {"Katılmıyorum": 1, "Kararsızım": 2, "Katılıyorum" : 3},
                    'PS2':  {"Katılmıyorum": 1, "Kararsızım": 2, "Katılıyorum" : 3},
                    'BPS1': {"Katılmıyorum": 1, "Kararsızım": 2, "Katılıyorum" : 3},
                    'BPS2': {"Katılmıyorum": 1, "Kararsızım": 2, "Katılıyorum" : 3},
                    'BPS4': {"Katılmıyorum": 1, "Kararsızım": 2, "Katılıyorum" : 3},
                    'BPS6': {"Katılmıyorum": 1, "Kararsızım": 2, "Katılıyorum" : 3}
                    }
    
    enc_df = df.replace(cleanup_nums) 
    return enc_df
    
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

def Train_Test_Splits(X_pca, df_Target):
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, df_Target, test_size=0.2, random_state=30)
    return X_test_pca, y_test

def Binary_Logistic_Regression(X_pca, df_Target):
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, df_Target, test_size=0.2, random_state=30)
    model_pca = LogisticRegression()
    model_pca.fit(X_train_pca, y_train.values.ravel())
    model_pca.score(X_test_pca, y_test)
    return model_pca


def Grid_Search_Hyper_Parameters(X_pca, df_Target):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, df_Target, test_size = 0.30, random_state = 42)
    
    xgboost_model = XGBClassifier()
    
    hyper_parameters = {
            'eta': [0.1, 0.01, 0.05],
            'max_depth': range(2, 10, 1)  
            }
    
    xgboost_cv = GridSearchCV(estimator = xgboost_model, param_grid = hyper_parameters, cv = 3, n_jobs = -1, verbose = 0)
    xgboost_cv.fit(X_train, y_train)    
    best = xgboost_cv.best_params_  
    return best

def XGBoost_Classification(X, df_Target, best):
    X_train, X_test, y_train, y_test = train_test_split(X, df_Target, test_size = 0.30, random_state = 42)
    clf = XGBClassifier()       
    clf.set_params(**best)
    xgb_tuned_model =  clf.fit(X_train,y_train.values.ravel())
    return xgb_tuned_model

def Random_Forest_Classification(X, df_Target):
    X_train, X_test, y_train, y_test = train_test_split(X, df_Target, test_size = 0.30, random_state = 42)
    model = RandomForestClassifier(n_estimators=15) # The n_estimators parameter was found 15 by trial and error in range(5-80).
    Random_model = model.fit(X_train, y_train.values.ravel())
    return Random_model

def Result_Output(Model,data, X, y, X_test, y_test):
    Predict = Model.predict(data)
    Prediction_Score = Model.score(X_test, y_test)
    
    if (Predict==1):
        result = "You are not anti-pandemic"
    else:
        result = "You are anti-pandemic"
    return result, Prediction_Score
 

   
st.write("""
         
         ## Anti-Pandemic Behaviour Prediction App
    This app predicts that are you anti-pandemic side or not  
         
         
     """)

st.sidebar.header('User Input Parameters')
st.sidebar.write(""" Demografik Bilgiler """)

def user_input_features():
    

    Cins = st.sidebar.selectbox('Cinsiyet',('Erkek','Kadın'))
    
    Yas = st.sidebar.selectbox('Yasiniz',("0-18","19-24", "25-34", "35-49","50-64","65+"))

    Egitim = st.sidebar.selectbox('Egitiminiz',("Ilkokul mezunu", "Ortaokul mezunu","Lise mezunu","Lisans mezunu","Yuksek Lisans mezunu","Doktora mezunu")) 
     
    Gelir = st.sidebar.selectbox('Geliriniz',("Ogrenciyim/ gecimimi baskasindan sagliyorum.", "0 -4250 tl", "4251- 6750 tl","6751 - 8250 tl", "8251 - 10750 tl", "10750 - 15000 tl", "15000tl+"))

    st.sidebar.write("""
                       
                     
                     Gorusler
                     
                     
                     """)

   
    GM4 = st.sidebar.selectbox('Pandemi ile ilgili dogru bilgileri geleneksel medya kanallarindan ogrendim.',('Evet','Hayır'))    
    CM4 = st.sidebar.selectbox('Pandemi ile ilgili dogru bilgileri cevremdeki kisilerden ogrendim.',('Evet','Hayır'))
    st.sidebar.write("""
                     
                     Geleneksel Kanallar: Televizyon,Gazete,Radio vb. 
                     
                     
                     Toplumsal Kanallar: Twitter, Facebook, Reddit vb.
                     
                     
                     Çevresel Kanallar: Aile,Arkadaş gibi Çevresel vb.
                     
                     """)  
                     
    MSS = st.sidebar.selectbox('Pandemi ile ilgili en güvenilir bilgiye hangi kanal uzerinden erisebildiginizi dusunuyorsunuz?',('Geleneksel Kanal','Toplumsal Kanal','Cevresel Kanal'))    
    RA1 = st.sidebar.selectbox('Pandemi boyunca, Yasadıgım bolgede coronavirus salgını cok ciddi seviyedeydi?',('Katılmıyorum','Kararsızım','Katılıyorum'))
    PS2 = st.sidebar.selectbox('Pandemi ile mücadelede aşının etkili oldugunu düsünüyorum.',('Katılmıyorum','Kararsızım','Katılıyorum'))    
    BPS1 = st.sidebar.selectbox('Pandemi boyunca toplum içinde daima maske taktim.',('Katılmıyorum','Kararsızım','Katılıyorum'))   
    BPS2 = st.sidebar.selectbox('Eve geldiginizde daima ellerimi en az 20sn. yıkadim.',('Katılmıyorum','Kararsızım','Katılıyorum'))    
    BPS4 = st.sidebar.selectbox('Pandemi ile ilgili dogru bildigim bilgileri baskalariyla da paylastim.',('Katılmıyorum','Kararsızım','Katılıyorum'))
    BPS6 = st.sidebar.selectbox('Pandemiyi onlemeye yonelik faaliyetlere gonullu katildim.',('Katılmıyorum','Kararsızım','Katılıyorum'))    

    Model_name = st.sidebar.selectbox('Tahminlemede kullanmak istediğiniz model hangisidir?',('Binary Logistic Regression', 'Random Forest Classification', 'XGBoost Classification'))         
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
    
    features = pd.DataFrame(data, index=[0],dtype=('string'))
    return features, Model_name, button
    
df, Model_name, button = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

def main():

    if (button):
        
        dataset = Read_csv('Anket_fin.csv')
        
        df_Ind, df_Target = Split_Column(dataset)
    
        enc_df = Encoding(df)
            
        df_Ind_merged = Merge_df(df_Ind, enc_df)
        
        X_pca, data = PCA_I(df_Ind_merged) 
        
        X_test, y_test = Train_Test_Splits(X_pca, df_Target)
        
        if (Model_name == "Binary Logistic Regression"):       
            Model = Binary_Logistic_Regression(X_pca, df_Target)
            
            
        elif (Model_name == "XGBoost Classification"):
            Best = Grid_Search_Hyper_Parameters(X_pca, df_Target)
            Model = XGBoost_Classification(X_pca, df_Target, Best)
            
        else:
            Model = Random_Forest_Classification(X_pca, df_Target)

        st.subheader('Class Labels')
        st.write("""
                1: You are not anti-pandemic
                
                0: You are anti-pandemic
            """)
      
        st.subheader('Prediction')
        
        result, prediction_score = Result_Output(Model,data, X_pca, df_Target, X_pca, df_Target)
            
        st.metric("",result,"Overall predictive score of "+str(Model_name)+": "+str(prediction_score))
 
        
    else:
        st.subheader('Please fill in the expressions on the left')
        st.write("""
               After filling out the expressions, simply press the Predict button.
            """)   

if __name__ == "__main__":
    main()
