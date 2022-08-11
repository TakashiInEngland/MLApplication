"""
@author: Takashi Ikeda
"""

# Feature Selection
def feature_selection(DF,i,target_var):
    
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.linear_model import LinearRegression
    #import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    
    X = DF.drop([target_var],axis=1).values
    y = DF[target_var].values
    y = y.reshape(-1, 1)
    l1 = DF.drop([target_var],axis=1).columns
    
    scX = preprocessing.StandardScaler()
    scaled_X = scX.fit_transform(X)
                                                
    scy = preprocessing.StandardScaler()
    scaled_y = scy.fit_transform(y)
    
    if i == "Linear Regressor":
        LR = LinearRegression()
        estimator = LR.fit(scaled_X,scaled_y)
        
        temp_dic = {}
        qwe = LR.coef_
        #qwe = qwe.tolist()
        for i in range(0,len(l1)):
            temp_dic[l1[i]] = qwe[:,i][0]
        
        temp_s = pd.Series(temp_dic)        
        score_result_LR = pd.DataFrame({"manipulated":temp_s.index, 
                                     "Coeff value LR":temp_s.values})
        score_result_LR = score_result_LR.set_index('manipulated')
        score_result_LR['Coeff value LR'] = score_result_LR['Coeff value LR'].abs()
        score_result_LR = score_result_LR.sort_values('Coeff value LR', ascending=False)
        #ss_LR = score_result_LR.index
        sr = score_result_LR
        
    elif i == "Ridge":
        RDG = Ridge(alpha=1.0)
        estimator = RDG.fit(scaled_X,scaled_y)
        
        temp_dic = {}
        qwe = RDG.coef_
        #qwe = qwe.tolist()
        for i in range(0,len(l1)):
            temp_dic[l1[i]] = qwe[:,i][0]
        
        temp_s = pd.Series(temp_dic)        
        score_result_RDG = pd.DataFrame({"manipulated":temp_s.index, 
                                     "Coeff value RDG":temp_s.values})
        score_result_RDG = score_result_RDG.set_index('manipulated')
        score_result_RDG['Coeff value RDG'] = score_result_RDG['Coeff value RDG'].abs()
        score_result_RDG = score_result_RDG.sort_values('Coeff value RDG', ascending=False)
        #ss_RDG = score_result_RDG.index
        sr = score_result_RDG
        
    elif i == "Lasso":
        LAS = Lasso(alpha=0.1)
        estimator = LAS.fit(scaled_X,scaled_y)
        
        temp_dic = {}
        qwe = LAS.coef_
        #qwe = qwe.tolist()
        for i in range(0,len(l1)):
            temp_dic[l1[i]] = qwe[i]
            
        temp_s = pd.Series(temp_dic)        
        score_result_LAS = pd.DataFrame({"manipulated":temp_s.index, 
                                     "Coeff value LAS":temp_s.values})
        score_result_LAS = score_result_LAS.set_index('manipulated')
        score_result_LAS['Coeff value LAS'] = score_result_LAS['Coeff value LAS'].abs()
        score_result_LAS = score_result_LAS.sort_values('Coeff value LAS', ascending=False)
        #ss_LAS = score_result_LAS.index
        sr = score_result_LAS
    
  #  elif i == "XGBoost":
   #     XGB = xgb.XGBRegressor()
    #    estimator = XGB.fit(scaled_X,scaled_y)
        
   #     temp_dic = {}
   #     qwe = XGB.feature_importances_
    #    #qwe = qwe.tolist()
    #    for i in range(0,len(l1)):
    #        temp_dic[l1[i]] = qwe[i]
        
    #    temp_s = pd.Series(temp_dic)        
    #    score_result_XGB = pd.DataFrame({"manipulated":temp_s.index, 
    #                                 "Coeff value XGB":temp_s.values})
    #    score_result_XGB = score_result_XGB.set_index('manipulated')
    #    score_result_XGB['Coeff value XGB'] = score_result_XGB['Coeff value XGB'].abs()
    #    score_result_XGB = score_result_XGB.sort_values('Coeff value XGB', ascending=False)
        #ss_XGB = score_result_XGB.index
    #    sr = score_result_XGB

    elif i == "Random Forest Regressor":
        RFR = RandomForestRegressor(max_depth = 2,n_estimators=300)
        estimator = RFR.fit(scaled_X,scaled_y)
        
        temp_dic = {}
        qwe = RFR.feature_importances_
        #qwe = qwe.tolist()
        for i in range(0,len(l1)):
            temp_dic[l1[i]] = qwe[i]
            
        temp_s = pd.Series(temp_dic)        
        score_result_RFR = pd.DataFrame({"manipulated":temp_s.index, 
                                     "Coeff value RFR":temp_s.values})
        score_result_RFR = score_result_RFR.set_index('manipulated')
        score_result_RFR['Coeff value RFR'] = score_result_RFR['Coeff value RFR'].abs()
        score_result_RFR = score_result_RFR.sort_values('Coeff value RFR', ascending=False)
        #ss_RFR = score_result_RFR.index
        sr = score_result_RFR
            
    elif i == "Gradient Boosting Regressor":
        GBR = GradientBoostingRegressor(n_estimators=100, 
                                          learning_rate=0.1,
                                          max_depth=1, 
                                          loss='ls')
        estimator = GBR.fit(scaled_X,scaled_y)
        
        temp_dic = {}
        qwe = GBR.feature_importances_
        #qwe = qwe.tolist()
        for i in range(0,len(l1)):
            temp_dic[l1[i]] = qwe[i]
        
        temp_s = pd.Series(temp_dic)        
        score_result_GBR = pd.DataFrame({"manipulated":temp_s.index, 
                                     "Coeff value GBR":temp_s.values})
        score_result_GBR = score_result_GBR.set_index('manipulated')
        score_result_GBR['Coeff value GBR'] = score_result_GBR['Coeff value GBR'].abs()
        score_result_GBR = score_result_GBR.sort_values('Coeff value GBR', ascending=False)
        #ss_GBR = score_result_GBR.index
        sr = score_result_GBR
        
    elif i == "Extra Tree Regressor":
        ETR = ExtraTreesRegressor()
        estimator = ETR.fit(scaled_X,scaled_y)
        
        temp_dic = {}
        qwe = ETR.feature_importances_
        #qwe = qwe.tolist()
        for i in range(0,len(l1)):
            temp_dic[l1[i]] = qwe[i]
        
        temp_s = pd.Series(temp_dic)        
        score_result_ETR = pd.DataFrame({"manipulated":temp_s.index, 
                                     "Coeff value ETR":temp_s.values})
        score_result_ETR = score_result_ETR.set_index('manipulated')
        score_result_ETR['Coeff value ETR'] = score_result_ETR['Coeff value ETR'].abs()
        score_result_ETR = score_result_ETR.sort_values('Coeff value ETR', ascending=False)
        #ss_ETR = score_result_ETR.index
        sr = score_result_ETR
    
    return sr