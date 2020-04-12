import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor





class house_prediction():
    def __init__(self,train,test):
        self.train=train
        self.test=test
        
    def summury_data(self):
        print("-------------------- Summury of Data --------------------------")
        data_head=self.train.head()
        print(data_head)
        decription=self.train.describe()
        print(decription)
        print(train.shape)
        data_info=self.train.info()
        print(data_info)
        print("------------------- TEst data---")
        data_info=self.test.info()
        print(data_info)
        
        train_null=self.train.isnull().sum()
        test_null=self.test.isnull().sum()
        print(train_null)
        print(test_null)


    def dealing_with_null(self):
        print("misiing value-------")
        self.test["MSZoning"].fillna("RL",inplace=True)
        self.test["SaleType"].fillna("Oth",inplace=True)
        self.test["Utilities"].fillna("AllPub",inplace=True)
        

        self.test["Exterior1st"].fillna("VinylSd",inplace=True)
        self.test["BsmtFinSF1"].fillna(0.0,inplace=True)
        self.test["BsmtFinSF2"].fillna(0.0,inplace=True)
        self.test["BsmtUnfSF"].fillna(0.0 ,inplace=True)
        self.test["TotalBsmtSF"].fillna(0.0,inplace=True)
        self.test["BsmtFullBath"].fillna(0.0 ,inplace=True)
        self.test["BsmtHalfBath"].fillna(0.0,inplace=True)
        self.test["KitchenQual"].fillna("TA",inplace=True)
        self.test["Functional"].fillna("Typ",inplace=True)
        self.test["GarageCars"].fillna(2.0 ,inplace=True)
        self.test["GarageArea"].fillna(0.0,inplace=True)
        self.test["Exterior2nd"].fillna("VinylSd",inplace=True)

     





        
        self.train["LotFrontage"].fillna(self.train.groupby("MSZoning")["LotFrontage"].transform("median"), inplace=True)
        self.test["LotFrontage"].fillna(self.test.groupby("MSZoning")["LotFrontage"].transform("median"), inplace=True)
        
        self.train["Alley"].fillna("None",inplace=True)
        self.test["Alley"].fillna("None",inplace=True)
        
        self.train["MasVnrType"].fillna("None",inplace=True)
        self.test["MasVnrType"].fillna("None",inplace=True)

    
        self.train["MasVnrArea"].fillna(self.train.groupby("MasVnrType")["MasVnrArea"].transform("median"), inplace=True)
        self.test["MasVnrArea"].fillna(self.test.groupby("MasVnrType")["MasVnrArea"].transform("median"), inplace=True)
        
        self.train["BsmtQual"].fillna("NA",inplace=True)
        self.test["BsmtQual"].fillna("NA",inplace=True)
        
        self.train["BsmtCond"].fillna("NA",inplace=True)
        self.test["BsmtCond"].fillna("NA",inplace=True)
        
        self.train["BsmtExposure"].fillna("NA",inplace=True)
        self.test["BsmtExposure"].fillna("NA",inplace=True)
        
        self.train["BsmtFinType1"].fillna("NA",inplace=True)
        self.test["BsmtFinType1"].fillna("NA",inplace=True)
        
        self.train["BsmtFinType2"].fillna("NA",inplace=True)
        self.test["BsmtFinType2"].fillna("NA",inplace=True)
        
        self.train["Electrical"].fillna("Mix",inplace=True)
        self.test["Electrical"].fillna("Mix",inplace=True)
        
        self.train["FireplaceQu"].fillna("NA",inplace=True)
        self.test["FireplaceQu"].fillna("NA",inplace=True)
        
        self.train["GarageType"].fillna("NA",inplace=True)
        self.test["GarageType"].fillna("NA",inplace=True)
                
        self.train["GarageFinish"].fillna("NA",inplace=True)
        self.test["GarageFinish"].fillna("NA",inplace=True)
        
        self.train["GarageQual"].fillna("NA",inplace=True)
        self.test["GarageQual"].fillna("NA",inplace=True)
        
        self.train["GarageCond"].fillna("NA",inplace=True)
        self.test["GarageCond"].fillna("NA",inplace=True)
        
        self.train["PoolQC"].fillna("NA",inplace=True)
        self.test["PoolQC"].fillna("NA",inplace=True)
        
        self.train["Fence"].fillna("NA",inplace=True)
        self.test["Fence"].fillna("NA",inplace=True)
        
        self.train["MiscFeature"].fillna("Othr",inplace=True)
        self.test["MiscFeature"].fillna("Othr",inplace=True)
        
        self.train["GarageYrBlt"].fillna(self.train["GarageYrBlt"].median( skipna = True) ,inplace=True)
        self.test["GarageYrBlt"].fillna(self.train["GarageYrBlt"].median( skipna = True) ,inplace=True)


    def mapping_data(self):
        map_data=[]
        column_name=[]
        map_dicts=[]
        for i in self.train:
            #print(i)
            if self.train[i].dtype ==object:
                column_name.append(i)
                map_data.append(self.train[i].unique())
       # print(map_data)
        
        for i in map_data:
            dict={}
            for j in range(len(i)):
                dict[i[j]]=j
            map_dicts.append(dict)
        #print(map_dicts)
        
        for k,m in zip(column_name,map_dicts):
            self.train[k]=self.train[k].map(m)
            self.test[k]=self.test[k].map(m)
            
            
    def modelling(self,predict_data):
        sample_submission=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/Data science data/predective analysis/kaggle comp/House Price Prediction/sample_submission.csv")

        self.train.drop('Id', axis=1, inplace=True)
        target=self.train["SalePrice"]
       # print(target)
        self.train.drop('SalePrice',axis=1,inplace=True)
        test_id=self.test["Id"]
        #print(test_id)
        self.test.drop('Id', axis=1, inplace=True)
        
      
        xg_reg = xgb.XGBRegressor(objective ='reg:linear',learning_rate = 0.2, colsample_bytree = 0.2
                                  ,max_depth = 7,alpha = 4, n_estimators = 59)
        
        xg_reg.fit(self.train,target)  
        print("------------------------ Prediction ------------------------------")
        preds = xg_reg.predict(self.test)
        print(preds)
        rmse = np.sqrt(mean_squared_error(sample_submission["SalePrice"], preds))
        print(rmse/1000)

        model =  RandomForestRegressor(max_depth=1)
        model.fit(self.train,target)      
        pred1=model.predict(self.test)
        print(pred1)
        rmse = np.sqrt(mean_squared_error(sample_submission["SalePrice"], pred1))
        print(rmse/1000)
        
        
        predict_data["Id"]=test_id
        predict_data["SalePrice"]=preds
        predict_data.to_csv('submission.csv',index=False) 
        
            
            
                

                
        
        

        
        
      
train=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/Data science data/predective analysis/kaggle comp/House Price Prediction/train.csv")
test=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/Data science data/predective analysis/kaggle comp/House Price Prediction/test.csv")
h=house_prediction(train,test)
h.summury_data()
predict_data=pd.DataFrame(
                          {'Id':[],
                           'SalePrice':[]
                           }
                    )
h.dealing_with_null()
h.summury_data()
h.mapping_data()
h.modelling(predict_data)
