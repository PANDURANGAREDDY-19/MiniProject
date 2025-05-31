def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Skip binary columns (like FastingBS and HeartDisease)
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 2]
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean