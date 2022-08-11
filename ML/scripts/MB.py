
def SP_model_building(DF,target_var,set_input_var,i):

    from sklearn import preprocessing
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    #import tensorflow as tf
    #from tensorflow.python.keras.layers import Input, Dense
    #from tensorflow.python.keras.models import Sequential
    
    import numpy as np
    np.random.seed(42)

    X = DF[set_input_var].values
    y = DF[target_var].values
    y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scX = preprocessing.StandardScaler()
    scaled_X = scX.fit_transform(X_train)
                                                
    scy = preprocessing.StandardScaler()
    scaled_y = scy.fit_transform(y_train)
    
    scaled_X_test = scX.transform(X_test)
    
    if i == "MLP":
        MLP = MLPRegressor(hidden_layer_sizes= (100,100,100,100,),random_state=42)
        MLP.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = MLP.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = MLP.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
    
    
    elif i == "LR":
        LR = LinearRegression()
        LR.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = LR.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = LR.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
    elif i == "XGB":
        XGB = xgb.XGBRegressor()
        XGB.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = XGB.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = XGB.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
    elif i == "RDG":
        RDG = Ridge(alpha=1.0)
        RDG.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = RDG.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = RDG.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
    elif i == "SVM":
        SVM = SVR()
        SVM.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = SVM.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = SVM.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
    elif i == "LAS":
        Lasso = Lasso(alpha=1.0)
        Lasso.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = Lasso.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = Lasso.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
    elif i == "GBR":
        GBR = GradientBoostingRegressor()
        GBR.fit(scaled_X,scaled_y)
        
        # prediction, using test data
        scaled_y_predicted = GBR.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = GBR.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
        
    elif i == "RFR":
        RFR = RandomForestRegressor()
        RFR.fit(scaled_X,scaled_y)
    
        # prediction, using test data
        scaled_y_predicted = RFR.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = RFR.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
    
    elif i == "POL":
        POL = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        POL.fit(scaled_X,scaled_y)
    
        # prediction, using test data
        scaled_y_predicted = POL.predict(scaled_X_test)
        unscaled_y_predicted = scy.inverse_transform(scaled_y_predicted)
        
        # prediction, using train data 
        scaled_y_train_predicted = POL.predict(scaled_X)
        unscaled_y_train_predicted = scy.inverse_transform(scaled_y_train_predicted)
    
    '''
        elif i == "KER":
        
        # Assign the number of columns
        n_cols = len(DF[set_input_var].columns)
        
        KER = Sequential()
            
        # Add the input layer
        KER.add(Dense(5,activation="relu",input_shape=(n_cols,)))
        
        # Add hideen layers
        KER.add(Dense(100,activation="relu"))
        KER.add(Dense(100,activation="relu"))
        KER.add(Dense(100,activation="relu"))
        
        # Add the output layer
        KER.add(Dense(1))
        
        # set up an optimiser
        optimiser = tf.keras.optimizers.RMSprop(0.001)
        
        # Compile the model
        KER.compile(optimizer=optimiser, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
        
        # Fit the model
        .fit(scaled_X,scaled_y)
        
     '''   
    return y_train,unscaled_y_train_predicted,y_test,unscaled_y_predicted
