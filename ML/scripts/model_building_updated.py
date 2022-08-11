
def model_building(DF,target_var,set_input_var):

    import warnings
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    # Fix the random state
    np.random.seed(42)
    
    # Import modules and models
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    ##### Feature Importance Visualisation ##### 
    kf = KFold(n_splits=5,random_state=42)
    
    X = DF[set_input_var].values
    y = DF[target_var].values
    y = y.reshape(-1, 1)
    
    #column_accumulated = []
    #countp = 0
    #DF_input_var = len(DF[set_input_var].columns)
    #for i in ss_XGB:      # Change the feasture selection results 
    #    column_accumulated.append(i)
    #    X = DF[column_accumulated].values
        
    #    countp += 1
        
    kf_ave_rmse_MLP = []
    kf_ave_rmse_LR = []
    kf_ave_rmse_XGB = []
    kf_ave_rmse_RDG = []
    kf_ave_rmse_SVM = []
    kf_ave_rmse_LAS = []
    kf_ave_rmse_GBR = []
    kf_ave_rmse_RFR = []
    kf_ave_rmse_POL = []
    
    train_MLP = []
    train_LR = []
    train_XGB = []
    train_RDG = []
    train_SVM = []
    train_LAS = []
    train_GBR = []
    train_RFR = []
    train_POL = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #y_train = y_train.ravel()
        #y_test = y_test.ravel()
        
        # Normalised data for model building
        scX = preprocessing.StandardScaler()
        scaled_X = scX.fit_transform(X_train)
                                                
        scy = preprocessing.StandardScaler()
        scaled_y = scy.fit_transform(y_train)
        
        scaled_X_test = scX.transform(X_test)
        
        ########## MLPRegressor ##########
        MLP = MLPRegressor(hidden_layer_sizes= (100,100,100,100,),random_state=42)
                                 #learning_rate_init = 0.2, 
                                 #tol = 0.05,
        MLP.fit(scaled_X,scaled_y)
        
        # prediction, using test data       
        scaled_y_predicted = MLP.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_MLP.append(rmse)
        
        
        # prediction, using train data 
        scaled_y_train_predicted = MLP.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
    
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_MLP.append(rmse_train)
    
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"MLP Regressor")
        ########## Linear Regressor ##########
        LR = LinearRegression()
                                 #learning_rate_init = 0.2, 
                                 #tol = 0.05,
    
        LR.fit(scaled_X,scaled_y)
        
        scaled_y_predicted = LR.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) /np.array(y_test))) * 100
        kf_ave_rmse_LR.append(rmse)
    
        scaled_y_train_predicted = LR.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_LR.append(rmse_train)
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"Linear Regressor")
            
        ########## XGBoost ##########
        XGB = xgb.XGBRegressor()
        XGB.fit(scaled_X,scaled_y)
        
        #plot_importance(scaled_nn)
        #pyplot.show()
        
        scaled_y_predicted = XGB.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_XGB.append(rmse)
        
        scaled_y_train_predicted = XGB.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_XGB.append(rmse_train)
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"XGB")
            
        ########## Ridge ##########
        RDG = Ridge(alpha=1.0,tol=0.0005)
                                 #learning_rate_init = 0.2, 
                                 #tol = 0.05,   
        RDG.fit(scaled_X,scaled_y)     

        nor_y_predicted = RDG.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(nor_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_RDG.append(rmse)
        
        scaled_y_train_predicted = RDG.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_RDG.append(rmse_train)
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"Ridge")
            
        ########## SVM ##########
        SVM = SVR()
    
        SVM.fit(scaled_X,scaled_y)
        
        scaled_y_predicted = SVM.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_SVM.append(rmse)
        
        scaled_y_train_predicted = SVM.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_SVM.append(rmse_train)
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"SVR")
            
        ########## Lasso ##########
        LASSO = Lasso(alpha=0.1)
    
        LASSO.fit(scaled_X,scaled_y)
        
        scaled_y_predicted = LASSO.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_LAS.append(rmse)
        
        scaled_y_train_predicted = LASSO.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_LAS.append(rmse_train)
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"Lasso")
            
        ########## Gradient Boosting Regressor ##########
        GBR = GradientBoostingRegressor(n_estimators=100, 
                                              learning_rate=0.1,
                                              max_depth=1, 
                                              loss='ls')
                                 #learning_rate_init = 0.2, 
                                 #tol = 0.05,
        GBR.fit(scaled_X,scaled_y)
        
        scaled_y_predicted = GBR.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_GBR.append(rmse)

        scaled_y_train_predicted = GBR.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_GBR.append(rmse_train)
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"GBR")
            
        ########## Random Forest Regressor ##########
        RFR = RandomForestRegressor(max_depth = 2,n_estimators=300)
                                 #learning_rate_init = 0.2, 
                                 #tol = 0.05,
    
        RFR.fit(scaled_X,scaled_y)

        scaled_y_predicted = RFR.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_RFR.append(rmse)
        
        scaled_y_train_predicted = RFR.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_RFR.append(rmse_train)
        
        ########## Polinomial Regressor ##########
        POL = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        POL.fit(scaled_X,scaled_y)
    
        scaled_y_predicted = POL.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        #rmse = sqrt(mean_squared_error(y_test,unscaled_y_predicted))
        rmse = np.mean(np.abs((np.array(y_test) - np.array(unscaled_y_predicted)) / np.array(y_test))) * 100
        kf_ave_rmse_POL.append(rmse)
        
        scaled_y_train_predicted = POL.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
        #rmse_train = sqrt(mean_squared_error(y_train,unscaled_y_train_predicted))
        rmse_train = np.mean(np.abs((np.array(y_train) - np.array(unscaled_y_train_predicted)) / np.array(y_train))) * 100
        train_POL.append(rmse_train)
        
        
        
        #if countp == DF_input_var:
        #    data_plot(y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted,"Random Forest Regressor")
            
    av_rmse_MLP = np.mean(kf_ave_rmse_MLP)
    av_rmse_LR = np.mean(kf_ave_rmse_LR)
    av_rmse_XGB = np.mean(kf_ave_rmse_XGB)
    av_rmse_RDG = np.mean(kf_ave_rmse_RDG)
    av_rmse_SVM = np.mean(kf_ave_rmse_SVM)
    av_rmse_LAS = np.mean(kf_ave_rmse_LAS)
    av_rmse_GBR = np.mean(kf_ave_rmse_GBR)
    av_rmse_RFR = np.mean(kf_ave_rmse_RFR)
    av_rmse_POL = np.mean(kf_ave_rmse_POL)
    
    av_train_MLP = np.mean(train_MLP)
    av_train_LR = np.mean(train_LR)
    av_train_XGB = np.mean(train_XGB)
    av_train_RDG = np.mean(train_RDG)
    av_train_SVM = np.mean(train_SVM)
    av_train_LAS = np.mean(train_LAS)
    av_train_GBR = np.mean(train_GBR)
    av_train_RFR = np.mean(train_RFR)
    av_train_POL = np.mean(train_POL)
    
    # Take a smaller RMSE through comparison
    RMSE_col = [["MLP",av_train_MLP,av_rmse_MLP],
                ["LR",av_train_LR,av_rmse_LR],
                ["XGB",av_train_XGB,av_rmse_XGB],
                ["RDG",av_train_RDG,av_rmse_RDG],
                ["SVM",av_train_SVM,av_rmse_SVM],
                ["LAS",av_train_LAS,av_rmse_LAS],
                ["GBR",av_train_GBR,av_rmse_GBR],
                ["RFR",av_train_RFR,av_rmse_RFR],
                ["POL",av_train_POL,av_rmse_POL]]
    RMSE_DF = pd.DataFrame(RMSE_col)
    RMSE_DF.columns = ["Algorithm","Average Train MAPE","Average Test MAPE"]
    
    #print("Average RMSEs for ", countp)
    #print("MLP",av_rmse_MLP)
    #print("LR",av_rmse_LR)
    #print("XGB",av_rmse_XGB)
    #print("Ridge",av_rmse_RDG)
    #print("SVR",av_rmse_SVM)
    #print("Lasso",av_rmse_LAS)
    #print("GBR",av_rmse_GBR)
    #print("RFR",av_rmse_RFR)
    
    Smallest_RMSE = RMSE_DF.iloc[:,1].min()
    Smallest_RMSE_Ind = RMSE_DF.iloc[:,1].idxmin()
    ALG_name = RMSE_DF.iloc[Smallest_RMSE_Ind,0]
    print("Smallest RMSE is ",Smallest_RMSE," achieved by ",ALG_name)
    print("------------------------------")
    return RMSE_DF
