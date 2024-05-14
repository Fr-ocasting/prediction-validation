import pandas as pd 


def find_nearest_date(date_series, date, inferior=True):
    """
    Find the nearest index of the timestamp <= or >= 'date' in 'date_series'.
    Parameters:
        date_series (pd.Series): A series of timestamps.
        date (pd.Timestamp): The reference timestamp.
        inferior (bool): If True, search for the nearest date <= 'date', else >= 'date'.
    Returns:
        int or None: The index of the nearest date, or None if not found.
    """
    # Calculating the difference
    diff = date_series - date
    
    if inferior:
        # Filtering to get the nearest <= date
        filtered_series = diff[diff <= pd.Timedelta(0)]
        if not filtered_series.empty:
            nearest_index = filtered_series.idxmax()
        else:
            return None,None
    else:
        # Filtering to get the nearest >= date
        filtered_series = diff[diff >= pd.Timedelta(0)]
        if not filtered_series.empty:
            nearest_index = filtered_series.idxmin()
        else:
            return None,None
    nearest_indice = date_series.index.get_loc(nearest_index)
    
    return nearest_index,nearest_indice


def find_limits_for_a_df(dataset,df_verif,predicted_serie,last_date1,prop,iteration):
    ''' Find the next limits of the df_valid or df_test '''
    #_,ind1 = find_nearest_date(df_verif,df_verif.iloc[:,-1],last_date1+dataset.shift_between_set,inferior = False)
    idx1,_ = find_nearest_date(predicted_serie,last_date1+dataset.shift_between_set,inferior = False)
    
    if (idx1 is None) : 
        return(None,None)
    elif (idx1 > predicted_serie.idxmax()):
        return(None,None) 
    else:
        first_date2 = predicted_serie.at[idx1]
        ind1= df_verif[f't+{dataset.step_ahead-1}'].index.get_loc(idx1)  # get indice from index
        ind2 = ind1+iteration*round(10*prop)
        if ind2 > len(df_verif) - 1:
            return(None,None)
        else:
            last_date2 = df_verif.iat[ind2,-1]
        
        return(first_date2,last_date2)


def train_valid_test_split_iterative_method(dataset,df_verif,train_prop,valid_prop,test_prop):
    # Init:
    # Case No Validation, No testing: 
    if train_prop == 1:
        return(df_verif[f't+{dataset.step_ahead-1}'].iat[0],df_verif[f't+{dataset.step_ahead-1}'].iat[-1],None,None,None,None)
    
    train_ind1 = 0
    train_ind2 = 0
    predicted_serie = df_verif[f't+{dataset.step_ahead-1}']
    first_train_date = predicted_serie.iat[train_ind1]
    last_train_date = predicted_serie.iat[train_ind2]
    new_first_valid_date,new_last_valid_date,new_first_test_date,new_last_test_date = 0,0,0,0 
    # Loop:
    iteration = 0
    while (new_first_valid_date is not None) | (new_first_test_date is not None):
        # Increment Train Limit
        iteration += 1
        new_train_ind2 = train_ind1 + iteration*round(10*train_prop)
        new_last_train_date = predicted_serie.iat[new_train_ind2]

        # Set Train,Valid,Test Limits:
        (new_first_valid_date,new_last_valid_date) = find_limits_for_a_df(dataset,df_verif,predicted_serie,new_last_train_date,valid_prop,iteration)
        if abs(test_prop)>1e-3:
            (new_first_test_date,new_last_test_date) = find_limits_for_a_df(dataset,df_verif,predicted_serie,new_last_valid_date,test_prop,iteration)
        # ...
            
        # Update 
        if (new_first_valid_date is not None) & (new_first_test_date is not None):
            first_valid_date,last_valid_date = new_first_valid_date,new_last_valid_date
            first_test_date,last_test_date = new_first_test_date,new_last_test_date
            train_ind2 = new_train_ind2
            last_train_date = new_last_train_date
        else:
            break


    # End Algorithm depending on wether test_set exists or not: 
    if abs(test_prop)<1e-3: 
        first_test_date,last_test_date = None,None
        last_valid_date = df_verif.iat[-1,-1]
    else: 
        test_ind2 = len(df_verif) - 1
        last_test_date = df_verif.iat[-1,-1]    
    # ==== ....

    split_limits = {'first_predicted_train_date':first_train_date,
                    'last_predicted_train_date': last_train_date,
                    'first_predicted_valid_date':first_valid_date,
                    'last_predicted_valid_date': last_valid_date,
                    'first_predicted_test_date': first_test_date,
                    'last_predicted_test_date': last_test_date
                    }

    return(split_limits)