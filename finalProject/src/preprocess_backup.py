

# preproess feature set X 
def preprocess_input(X, le0, le1, le2, oHot, supplemental_label_data=pd.DataFrame()):
    
    # drop unwanted features
    
    unwanted_columns = ["playerName", "Season", "spacer1", "transferredSchools", "ORB", "DRB"]  # "\xa0"
    X = X.drop(unwanted_columns, axis=1)

    
    # organize features
    columns = list(X.columns)
    categorical = ["position", "School", "Conf"]
    #diff = lambda l1, l2: [x for x in l1 if x not in l2]
    #numerical = diff(columns, categorical) 
    
    
    # cast numerical data as floats
    for col in columns:
        if col not in categorical:
            X[col] = X[col].astype(float)
    
    
    
    # impute: fill in missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = X.values # convert to ndarray of objects 
    imputer = imputer.fit(X[:, 3:])
    X[:, 3:] = imputer.transform(X[:, 3:])
    
    
    
    # scale data #
    X_scaler = StandardScaler()
    X_scaler = X_scaler.fit(X[:, 3:])
    X[:, 3:] = X_scaler.transform(X[:, 3:])

    
    
    
    # encode categorial data #
    
    fit_data0 = X[:, 0]
    fit_data1 = X[:, 1]
    fit_data2 = X[:, 2]
    print(fit_data0)
    print(fit_data0.shape)
        
    if len(supplemental_label_data) > 0:
        
        fit_data0 = list(fit_data0)
        fit_data0.extend(list(supplemental_label_data.School))
        
        print("yale:",fit_data0.index('Yale'))
        fit_data1 = list(fit_data1)
        fit_data1.extend(list(supplemental_label_data.Conf))
        
        fit_data2 = list(fit_data2)
        fit_data2.extend(list(supplemental_label_data.position))

        
    if not le0:
        # label encode
        labelEncoder_X0 = LabelEncoder()
        labelEncoder_X0 = labelEncoder_X0.fit(fit_data0)
        X[:, 0] = labelEncoder_X0.transform(X[:, 0])
        
        labelEncoder_X1 = LabelEncoder()
        labelEncoder_X1 = labelEncoder_X1.fit(fit_data1)
        X[:, 1] = labelEncoder_X1.transform(X[:, 1])
        
        labelEncoder_X2 = LabelEncoder()
        labelEncoder_X2 = labelEncoder_X2.fit(fit_data2)
        X[:, 2] = labelEncoder_X2.transform(X[:, 2])
        
        # one hot encode
        oneHotEncoder = OneHotEncoder(categorical_features=[0,1,2], handle_unknown='ignore')
        oneHotEncoder = oneHotEncoder.fit(X)
        X = oneHotEncoder.transform(X)
    else:
        # label encoder with prev encoder
        X[:, 0] = le0.transform(X[:, 0])
        X[:, 1] = le1.transform(X[:, 1])
        X[:, 2] = le2.transform(X[:, 2])
        
        # one hot encode with prev encoder
        X = oHot.transform(X)
        
        return X
        
    