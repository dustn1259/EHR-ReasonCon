import pandas as pd
def search_table(df1, df2, table1, entity=None, search_type=None, comparison_type=None, 
                valuenum=None, valuenum2=None, date1=None, date2=None, time_expression=None,category = None):
    if valuenum is not None:
        try:
            valuenum = float(valuenum)
        except (ValueError, TypeError):
            valuenum = None
    
    # process valuenum2
    if valuenum2 is not None:
        try:
            valuenum2 = float(valuenum2)
        except (ValueError, TypeError):
            valuenum2 = None
    """
    Unified function to handle all types of searches in MIMIC tables
    
    Args:
        df1: Main dataframe to search in
        df2: Dictionary/reference dataframe
        table1: Table name being searched
        entity: Entity/drug name to search for (optional)
        search_type: 'entity_only', 'entity_value', or 'value_only'
        comparison_type: 'equal', 'more', 'less', or 'between' (required if search_type involves values)
        valuenum: Value to compare against
        valuenum2: Second value for 'between' comparison
    """
    def process_date_range(df, table1, date_col, time_expression, date1, date2=None):
        import datetime
        
        # First, ensure the date column is datetime type
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                raise ValueError(f"Column {date_col} could not be converted to datetime: {str(e)}")
        
        # convert date1 to a datetime object
        if not isinstance(date1, pd.Timestamp) and not isinstance(date1, datetime.datetime):
            date1_dt = pd.to_datetime(date1)
        else:
            date1_dt = date1
        
        if date2 and not isinstance(date2, pd.Timestamp) and not isinstance(date2, datetime.datetime):
            date2_dt = pd.to_datetime(date2)
        else:
            date2_dt = date2
        
        # Check whether time information exists (if hour, minute, and second are not all 0, consider that time info exists)
        has_time = (date1_dt.hour != 0 or date1_dt.minute != 0 or date1_dt.second != 0)
        if time_expression == '1':
            # For tables with start/end times
            if table1 == 'prescriptions':
                start_col = 'STARTDATE'
                end_col = 'ENDDATE'
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])
                
                if has_time:
                    # Use exact timestamp
                    mask = (df[start_col] <= date1_dt) & (df[end_col] >= date1_dt)
                else:
                    # Use just the date part
                    target_date = date1_dt.date()
                    mask = (df[start_col].dt.date <= target_date) & (df[end_col].dt.date >= target_date)
            
            elif (table1 == 'inputevents_mv' or table1 == 'procedureevents_mv'):
                start_col = 'STARTTIME'
                end_col = 'ENDTIME'
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])
                
                if has_time:
                    # Use exact timestamp
                    mask = (df[start_col] <= date1_dt) & (df[end_col] >= date1_dt)
                else:
                    # Use just the date part
                    target_date = date1_dt.date()
                    mask = (df[start_col].dt.date <= target_date) & (df[end_col].dt.date >= target_date)
            else:
                # For tables with single timestamp
                if has_time:
                    # Compare with full timestamp
                    mask = df[date_col] == date1_dt
                else:
                    # Compare only the date part
                    mask = df[date_col].dt.date == date1_dt.date()
                
        elif time_expression == '2':
            base_date = date1_dt.date()
            date_start = base_date - pd.Timedelta(days=1)
            date_end = base_date + pd.Timedelta(days=1)
            
            if table1 == 'prescriptions':
                start_col = 'STARTDATE'
                end_col = 'ENDDATE'
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])

                mask = (df[start_col].dt.date <= date_end) & (df[end_col].dt.date >= date_start)
                
            elif (table1 == 'inputevents_mv' or table1 == 'procedureevents_mv'):
                start_col = 'STARTTIME'
                end_col = 'ENDTIME'
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])

                mask = (df[start_col].dt.date <= date_end) & (df[end_col].dt.date >= date_start)
        
            else:
                mask = (df[date_col].dt.date >= date_start) & (df[date_col].dt.date <= date_end)
            
        elif time_expression == '3':
            if not isinstance(date2, pd.Timestamp) and not isinstance(date2, datetime.datetime):
                date2_dt = pd.to_datetime(date2)
            else:
                date2_dt = date2
                
            admission_date = date1_dt.date() - pd.Timedelta(days=1)
            discharge_date = date2_dt.date() + pd.Timedelta(days=1)
            
            if table1 == 'prescriptions':
                start_col = 'STARTDATE'
                end_col = 'ENDDATE'
                
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])
                
                mask = (df[end_col].dt.date >= admission_date) & (df[start_col].dt.date <= discharge_date)
            
            elif (table1 == 'inputevents_mv' or table1 == 'procedureevents_mv'):
                start_col = 'STARTTIME'
                end_col = 'ENDTIME'
                
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])

                mask = (df[end_col].dt.date >= admission_date) & (df[start_col].dt.date <= discharge_date)

            else:
                mask = ((df[date_col].dt.date >= admission_date) & 
                    (df[date_col].dt.date <= discharge_date))


        elif time_expression == '4':
            date_start = date1_dt.date()
            date_end = date2_dt.date()
            
            if table1 == 'prescriptions':
                start_col = 'STARTDATE'
                end_col = 'ENDDATE'
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])

                mask = (df[start_col].dt.date <= date_end) & (df[end_col].dt.date >= date_start)
                
            elif (table1 == 'inputevents_mv' or table1 == 'procedureevents_mv'):
                start_col = 'STARTTIME'
                end_col = 'ENDTIME'
                if not pd.api.types.is_datetime64_any_dtype(df[start_col]):
                    df[start_col] = pd.to_datetime(df[start_col])
                if not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    df[end_col] = pd.to_datetime(df[end_col])

                mask = (df[start_col].dt.date <= date_end) & (df[end_col].dt.date >= date_start)
        
            else:
                mask = (df[date_col].dt.date >= date_start) & (df[date_col].dt.date <= date_end)
        return mask

    if search_type == 'category_date':
        if table1 == 'prescriptions':
            # For prescriptions, directly filter by DRUG column
            print('there is no category')
        else:
            # For other tables, use the dictionary mapping
            if len(df2[df2['CATEGORY'] == category]['ITEMID']) == 1:
                itemid_value_item = df2[df2['CATEGORY'] == category]['ITEMID'].iloc[0]
                if table1 == 'microbiologyevents':
                    merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                    merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                    merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
                    base_result = merged_df[(merged_df['SPEC_ITEMID'] == itemid_value_item) | (merged_df['ORG_ITEMID'] == itemid_value_item)]
                else:
                    merged_df = df1.merge(df2, on='ITEMID', how='left')
                    base_result = merged_df[merged_df['ITEMID'] == itemid_value_item]
                date_col = 'CHARTTIME' if table1 not in ['inputevents_mv', 'procedureevents_mv'] else 'STARTTIME'
                if table1 == 'microbiologyevents':
                    date_col = 'CHARTDATE'
            else:
                results = []
                for itemid_value_item in df2[df2['CATEGORY'] == category]['ITEMID']:
                    if table1 == 'microbiologyevents':
                        merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                        merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                        merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
                        temp_result = merged_df[(merged_df['SPEC_ITEMID'] == itemid_value_item) | (merged_df['ORG_ITEMID'] == itemid_value_item)]
                    else:
                        merged_df = df1.merge(df2, on='ITEMID', how='left')
                        temp_result = merged_df[merged_df['ITEMID'] == itemid_value_item]
                    results.append(temp_result)
                base_result = pd.concat(results) if results else pd.DataFrame()
                date_col = 'CHARTTIME' if table1 not in ['inputevents_mv','procecdureevents_mv'] else 'STARTTIME'
                if table1 == 'microbiologyevents':
                    date_col = 'CHARTDATE'
        if len(base_result.columns) == 0:
            result = base_result
        else:
            date_mask = process_date_range(base_result, table1, date_col, time_expression, date1, date2)
            result = base_result[date_mask]

    elif search_type == 'entity_date':
        if table1 == 'prescriptions':
            # For prescriptions, directly filter by DRUG column
            base_result = df1[df1['DRUG'] == entity]
            date_col = 'STARTDATE'
        else:
            # For other tables, use the dictionary mapping
            if len(df2[df2['LABEL'] == entity]['ITEMID']) == 1:
                itemid_value_item = df2[df2['LABEL'] == entity]['ITEMID'].iloc[0]
                if table1 == 'microbiologyevents':
                    merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                    merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                    merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
                    base_result = merged_df[(merged_df['SPEC_ITEMID'] == itemid_value_item) | (merged_df['ORG_ITEMID'] == itemid_value_item)]
                else:
                    merged_df = df1.merge(df2, on='ITEMID', how='left')
                    base_result = merged_df[merged_df['ITEMID'] == itemid_value_item]
                date_col = 'CHARTTIME' if table1 not in ['inputevents_mv', 'procedureevents_mv'] else 'STARTTIME'
                if table1 == 'microbiologyevents':
                    date_col = 'CHARTDATE'
            else:
                results = []
                for itemid_value_item in df2[df2['LABEL'] == entity]['ITEMID']:
                    if table1 == 'microbiologyevents':
                        merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                        merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                        merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
                        temp_result = merged_df[(merged_df['SPEC_ITEMID'] == itemid_value_item) | (merged_df['ORG_ITEMID'] == itemid_value_item)]
                    else:
                        merged_df = df1.merge(df2, on='ITEMID', how='left')
                        temp_result = merged_df[merged_df['ITEMID'] == itemid_value_item]
                    results.append(temp_result)
                base_result = pd.concat(results) if results else pd.DataFrame()
                date_col = 'CHARTTIME' if table1 not in ['inputevents_mv','procedureevents_mv'] else 'STARTTIME'
                if table1 == 'microbiologyevents':
                    date_col = 'CHARTDATE'


        if len(base_result.columns) == 0:
            result = base_result
        else:
            date_mask = process_date_range(base_result, table1, date_col, time_expression, date1, date2)
            result = base_result[date_mask]
    # Date-value search
    elif search_type == 'date_value':
        if table1 == 'prescriptions':
            # First filter by date
            date_col = 'STARTDATE'
            if len(df1.columns) == 0:
                date_filtered = df1
            else:
                date_mask = process_date_range(df1, table1, date_col, time_expression, date1, date2)
                date_filtered = df1[date_mask]
            
            # Then filter by value
            date_filtered['DOSE_VAL_RX'] = pd.to_numeric(date_filtered['DOSE_VAL_RX'], errors='coerce')
            if comparison_type == 'equal':
                result = date_filtered[date_filtered['DOSE_VAL_RX'] == valuenum]
            elif comparison_type == 'more':
                result = date_filtered[date_filtered['DOSE_VAL_RX'] > valuenum]
            elif comparison_type == 'less':
                result = date_filtered[date_filtered['DOSE_VAL_RX'] < valuenum]
            elif comparison_type == 'between':
                result = date_filtered[(date_filtered['DOSE_VAL_RX'] >= valuenum) & 
                            (date_filtered['DOSE_VAL_RX'] <= valuenum2)]
        else:
            # For other tables, first merge with dictionary
            if table1 == 'microbiologyevents':
                    merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                    merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                    merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
            else:
                merged_df = df1.merge(df2, on='ITEMID', how='left')
            
            # Filter by date
            date_col = 'CHARTTIME' if table1 not in ['inputevents_mv','procedureevents_mv'] else 'STARTTIME'
            if table1 == 'microbiologyevents':
                date_col = 'CHARTDATE'
            if len(merged_df.columns) == 0:
                date_filtered = merged_df
            else:
                date_mask = process_date_range(merged_df, table1, date_col, time_expression, date1, date2)
                date_filtered = merged_df[date_mask]
            
            # Then filter by value
            if table1.lower() in ['inputevents_mv', 'inputevents_cv']:
                if comparison_type == 'equal':
                    lower_bound = valuenum - 0.01
                    upper_bound = valuenum + 0.01
                    result = date_filtered[
                        ((date_filtered['RATE'] >= lower_bound) & (date_filtered['RATE'] <= upper_bound)) |
                        ((date_filtered['AMOUNT'] >= lower_bound) & (date_filtered['AMOUNT'] <= upper_bound)) |
                        ((date_filtered['ORIGINALRATE'] >= lower_bound) & (date_filtered['ORIGINALRATE'] <= upper_bound)) |
                        ((date_filtered['ORIGINALAMOUNT'] >= lower_bound) & (date_filtered['ORIGINALAMOUNT'] <= upper_bound))
                    ]
                elif comparison_type == 'more':
                    result = date_filtered[
                        (date_filtered['RATE'] > valuenum) |
                        (date_filtered['AMOUNT'] > valuenum) |
                        (date_filtered['ORIGINALRATE'] > valuenum) |
                        (date_filtered['ORIGINALAMOUNT'] > valuenum)
                    ]
                elif comparison_type == 'less':
                    result = date_filtered[
                        (date_filtered['RATE'] < valuenum) |
                        (date_filtered['AMOUNT'] < valuenum) |
                        (date_filtered['ORIGINALRATE'] < valuenum) |
                        (date_filtered['ORIGINALAMOUNT'] < valuenum)
                    ]
                elif comparison_type == 'between':
                    result = date_filtered[
                        ((date_filtered['RATE'] >= valuenum) & (date_filtered['RATE'] <= valuenum2)) |
                        ((date_filtered['AMOUNT'] >= valuenum) & (date_filtered['AMOUNT'] <= valuenum2)) |
                        ((date_filtered['ORIGINALRATE'] >= valuenum) & (date_filtered['ORIGINALRATE'] <= valuenum2)) |
                        ((date_filtered['ORIGINALAMOUNT'] >= valuenum) & (date_filtered['ORIGINALAMOUNT'] <= valuenum2))
                    ]
            else:
                if comparison_type == 'equal':
                    result = date_filtered[date_filtered['VALUENUM'] == valuenum]
                elif comparison_type == 'more':
                    result = date_filtered[date_filtered['VALUENUM'] > valuenum]
                elif comparison_type == 'less':
                    result = date_filtered[date_filtered['VALUENUM'] < valuenum]
                elif comparison_type == 'between':
                    result = date_filtered[
                        (date_filtered['VALUENUM'] >= valuenum) & 
                        (date_filtered['VALUENUM'] <= valuenum2)
                    ]


    elif search_type == 'date_only':
        if table1 == 'prescriptions':
            base_result = df1  # prescriptions do not need merging
            date_col = 'STARTDATE'
        else:
            # other tables need to merge df1 and df2
            base_result = df1.merge(df2, on='ITEMID', how='left')
            date_col = 'CHARTTIME' if table1 not in ['inputevents_mv','procedureevents_mv'] else 'STARTTIME'
            if table1 == 'microbiologyevents':
                date_col = 'CHARTDATE'
        
        if len(base_result.columns) == 0:
            result = base_result
        else:
            date_mask = process_date_range(base_result, table1, date_col, time_expression, date1, date2)
            result = base_result[date_mask]

        

        # Entity-value-date search
    elif search_type == 'entity_value_date':
        if table1 == 'prescriptions':
            # First filter by entity (drug name)
            base_result = df1[df1['DRUG'] == entity]
            
            # Then filter by value
            base_result['DOSE_VAL_RX'] = pd.to_numeric(base_result['DOSE_VAL_RX'], errors='coerce')
            if comparison_type == 'equal':
                value_filtered = base_result[base_result['DOSE_VAL_RX'] == valuenum]
            elif comparison_type == 'more':
                value_filtered = base_result[base_result['DOSE_VAL_RX'] > valuenum]
            elif comparison_type == 'less':
                value_filtered = base_result[base_result['DOSE_VAL_RX'] < valuenum]
            elif comparison_type == 'between':
                value_filtered = base_result[(base_result['DOSE_VAL_RX'] >= valuenum) & 
                        (base_result['DOSE_VAL_RX'] <= valuenum2)]
            
            # Finally filter by date
            date_col = 'STARTDATE'
            if len(value_filtered.columns) == 0:
                result = value_filtered
            else:
                date_mask = process_date_range(value_filtered, table1, date_col, time_expression, date1, date2)
                result = value_filtered[date_mask]
        else:
            # For other tables, first get the entity by ITEMID from dictionary
            if len(df2[df2['LABEL'] == entity]['ITEMID']) == 1:
                # Single item case
                itemid_value_item = df2[df2['LABEL'] == entity]['ITEMID'].iloc[0]
                if table1 == 'microbiologyevents':
                    merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                    merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                    merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
                    base_result=merged_df[(merged_df['SPEC_ITEMID'] == itemid_value_item) | (merged_df['ORG_ITEMID'] == itemid_value_item)]
                else:
                    merged_df = df1.merge(df2, on='ITEMID', how='left')
                    base_result = merged_df[merged_df['ITEMID'] == itemid_value_item]
            else:
                # Multiple items case
                results = []
                for itemid_value_item in df2[df2['LABEL'] == entity]['ITEMID']:
                    if table1 == 'microbiologyevents':
                        merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                        merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                        merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
                        temp_result = merged_df[(merged_df['SPEC_ITEMID'] == itemid_value_item) | (merged_df['ORG_ITEMID'] == itemid_value_item)]
                    else:
                        merged_df = df1.merge(df2, on='ITEMID', how='left')
                        temp_result = merged_df[merged_df['ITEMID'] == itemid_value_item]
                    results.append(temp_result)
                base_result = pd.concat(results) if results else pd.DataFrame()
            
            # Then filter by value
            if table1.lower() in ['inputevents_mv', 'inputevents_cv']:
                if comparison_type == 'equal':
                    lower_bound = valuenum - 0.01
                    upper_bound = valuenum + 0.01
                    value_filtered = base_result[
                        ((base_result['RATE'] >= lower_bound) & (base_result['RATE'] <= upper_bound)) |
                        ((base_result['AMOUNT'] >= lower_bound) & (base_result['AMOUNT'] <= upper_bound)) |
                        ((base_result['ORIGINALRATE'] >= lower_bound) & (base_result['ORIGINALRATE'] <= upper_bound)) |
                        ((base_result['ORIGINALAMOUNT'] >= lower_bound) & (base_result['ORIGINALAMOUNT'] <= upper_bound))
                    ]
                elif comparison_type == 'more':
                    value_filtered = base_result[
                        (base_result['RATE'] > valuenum) |
                        (base_result['AMOUNT'] > valuenum) |
                        (base_result['ORIGINALRATE'] > valuenum) |
                        (base_result['ORIGINALAMOUNT'] > valuenum)
                    ]
                elif comparison_type == 'less':
                    value_filtered = base_result[
                        (base_result['RATE'] < valuenum) |
                        (base_result['AMOUNT'] < valuenum) |
                        (base_result['ORIGINALRATE'] < valuenum) |
                        (base_result['ORIGINALAMOUNT'] < valuenum)
                    ]
                elif comparison_type == 'between':
                    value_filtered = base_result[
                        ((base_result['RATE'] >= valuenum) & (base_result['RATE'] <= valuenum2)) |
                        ((base_result['AMOUNT'] >= valuenum) & (base_result['AMOUNT'] <= valuenum2)) |
                        ((base_result['ORIGINALRATE'] >= valuenum) & (base_result['ORIGINALRATE'] <= valuenum2)) |
                        ((base_result['ORIGINALAMOUNT'] >= valuenum) & (base_result['ORIGINALAMOUNT'] <= valuenum2))
                    ]
            else:
                if comparison_type == 'equal':
                    value_filtered = base_result[base_result['VALUENUM'] == valuenum]
                elif comparison_type == 'more':
                    value_filtered = base_result[base_result['VALUENUM'] > valuenum]
                elif comparison_type == 'less':
                    value_filtered = base_result[base_result['VALUENUM'] < valuenum]
                elif comparison_type == 'between':
                    value_filtered = base_result[
                        (base_result['VALUENUM'] >= valuenum) & 
                        (base_result['VALUENUM'] <= valuenum2)
                    ]
            
            # Finally filter by date
            date_col = 'CHARTTIME' if table1 not in ['inputevents_mv','procedureevents_mv'] else 'STARTTIME'
            if table1 == 'microbiologyevents':
                date_col = 'CHARTDATE'
            if len(value_filtered.columns) == 0:
                result = value_filtered
            else:
                date_mask = process_date_range(value_filtered, table1, date_col, time_expression, date1, date2)
                result = value_filtered[date_mask]
    

    # Entity only search
    elif search_type == 'entity_only':
        if table1 == 'prescriptions':
            result = df1[df1['DRUG'] == entity]
        else:
            if len(df2[df2['LABEL'] == entity]['ITEMID']) == 1:
                itemid_value_item = df2[df2['LABEL'] == entity]['ITEMID'].iloc[0]
                result = df1.merge(df2, on='ITEMID', how='left')[
                    (df1['ITEMID'] == itemid_value_item)]
            else:
                results = []
                for itemid_value_item in df2[df2['LABEL'] == entity]["ITEMID"]:
                    temp_result = df1.merge(df2, on='ITEMID', how='left')[
                        (df1['ITEMID'] == itemid_value_item)]
                    results.append(temp_result)
                result = pd.concat(results) if results else pd.DataFrame()
    
    # Value only search
    elif search_type == 'value_only':
        if table1 == 'prescriptions':
            base_result = df1
            base_result['DOSE_VAL_RX'] = pd.to_numeric(base_result['DOSE_VAL_RX'], errors='coerce')
            if comparison_type == 'equal':
                result = base_result[base_result['DOSE_VAL_RX'] == valuenum]
            elif comparison_type == 'more':
                result = base_result[base_result['DOSE_VAL_RX'] > valuenum]
            elif comparison_type == 'less':
                result = base_result[base_result['DOSE_VAL_RX'] < valuenum]
            elif comparison_type == 'between':
                result = base_result[(base_result['DOSE_VAL_RX'] >= valuenum) & 
                           (base_result['DOSE_VAL_RX'] <= valuenum2)]
        else:
            if table1 == 'microbiologyevents':
                    merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                    merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                    merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
            else:
                merged_df = df1.merge(df2, on='ITEMID', how='left')
            if table1.lower() in ['inputevents_mv', 'inputevents_cv']:
                if comparison_type == 'equal':
                    lower_bound = valuenum - 0.01
                    upper_bound = valuenum + 0.01
                    result = merged_df[
                        ((merged_df['RATE'] >= lower_bound) & (merged_df['RATE'] <= upper_bound)) |
                        ((merged_df['AMOUNT'] >= lower_bound) & (merged_df['AMOUNT'] <= upper_bound)) |
                        ((merged_df['ORIGINALRATE'] >= lower_bound) & (merged_df['ORIGINALRATE'] <= upper_bound)) |
                        ((merged_df['ORIGINALAMOUNT'] >= lower_bound) & (merged_df['ORIGINALAMOUNT'] <= upper_bound))
                    ]
                elif comparison_type == 'more':
                    result = merged_df[
                        (merged_df['RATE'] > valuenum) |
                        (merged_df['AMOUNT'] > valuenum) |
                        (merged_df['ORIGINALRATE'] > valuenum) |
                        (merged_df['ORIGINALAMOUNT'] > valuenum)
                    ]
                elif comparison_type == 'less':
                    result = merged_df[
                        (merged_df['RATE'] < valuenum) |
                        (merged_df['AMOUNT'] < valuenum) |
                        (merged_df['ORIGINALRATE'] < valuenum) |
                        (merged_df['ORIGINALAMOUNT'] < valuenum)
                    ]
                elif comparison_type == 'between':
                    result = merged_df[
                        ((merged_df['RATE'] >= valuenum) & (merged_df['RATE'] <= valuenum2)) |
                        ((merged_df['AMOUNT'] >= valuenum) & (merged_df['AMOUNT'] <= valuenum2)) |
                        ((merged_df['ORIGINALRATE'] >= valuenum) & (merged_df['ORIGINALRATE'] <= valuenum2)) |
                        ((merged_df['ORIGINALAMOUNT'] >= valuenum) & (merged_df['ORIGINALAMOUNT'] <= valuenum2))
                    ]
            else:
                if comparison_type == 'equal':
                    result = merged_df[merged_df['VALUENUM'] == valuenum]
                elif comparison_type == 'more':
                    result = merged_df[merged_df['VALUENUM'] > valuenum]
                elif comparison_type == 'less':
                    result = merged_df[merged_df['VALUENUM'] < valuenum]
                elif comparison_type == 'between':
                    result = merged_df[
                        (merged_df['VALUENUM'] >= valuenum) & 
                        (merged_df['VALUENUM'] <= valuenum2)
                    ]
    
    # Entity and value combined search
    elif search_type == 'entity_value':
        if table1 == 'prescriptions':
            base_result = df1[df1['DRUG'] == entity]
            base_result['DOSE_VAL_RX'] = pd.to_numeric(base_result['DOSE_VAL_RX'], errors='coerce')
            if comparison_type == 'equal':
                result = base_result[base_result['DOSE_VAL_RX'] == valuenum]
            elif comparison_type == 'more':
                result = base_result[base_result['DOSE_VAL_RX'] > valuenum]
            elif comparison_type == 'less':
                result = base_result[base_result['DOSE_VAL_RX'] < valuenum]
            elif comparison_type == 'between':
                result = base_result[
                    (base_result['DOSE_VAL_RX'] >= valuenum) & 
                    (base_result['DOSE_VAL_RX'] <= valuenum2)
                ]
        else:
            itemid_value_item = df2[df2['LABEL'] == entity]['ITEMID'].iloc[0]
            if table1 == 'microbiologyevents':
                    merged_df = pd.merge(df1, df2, left_on='SPEC_ITEMID', right_on='ITEMID', how='left')
                    merged_df = pd.merge(merged_df, df2, left_on='ORG_ITEMID', right_on='ITEMID', how='left')
                    merged_df=merged_df[['ROW_ID_x', 'HADM_ID', 'CHARTDATE', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'ORG_ITEMID', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION','LABEL_x', 'LINKSTO_x', 'CATEGORY_x','LABEL_y', 'LINKSTO_y', 'CATEGORY_y']]
            else:
                merged_df = df1.merge(df2, on='ITEMID', how='left')
            
            if table1.lower() in ['inputevents_mv', 'inputevents_cv']:
                if comparison_type == 'equal':
                    lower_bound = valuenum - 0.01
                    upper_bound = valuenum + 0.01
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (
                            ((merged_df['RATE'] >= lower_bound) & (merged_df['RATE'] <= upper_bound)) |
                            ((merged_df['AMOUNT'] >= lower_bound) & (merged_df['AMOUNT'] <= upper_bound)) |
                            ((merged_df['ORIGINALRATE'] >= lower_bound) & (merged_df['ORIGINALRATE'] <= upper_bound)) |
                            ((merged_df['ORIGINALAMOUNT'] >= lower_bound) & (merged_df['ORIGINALAMOUNT'] <= upper_bound))
                        )
                    ]
                elif comparison_type == 'more':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (
                            (merged_df['RATE'] > valuenum) |
                            (merged_df['AMOUNT'] > valuenum) |
                            (merged_df['ORIGINALRATE'] > valuenum) |
                            (merged_df['ORIGINALAMOUNT'] > valuenum)
                        )
                    ]
                elif comparison_type == 'less':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (
                            (merged_df['RATE'] < valuenum) |
                            (merged_df['AMOUNT'] < valuenum) |
                            (merged_df['ORIGINALRATE'] < valuenum) |
                            (merged_df['ORIGINALAMOUNT'] < valuenum)
                        )
                    ]
                elif comparison_type == 'between':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (
                            ((merged_df['RATE'] >= valuenum) & (merged_df['RATE'] <= valuenum2)) |
                            ((merged_df['AMOUNT'] >= valuenum) & (merged_df['AMOUNT'] <= valuenum2)) |
                            ((merged_df['ORIGINALRATE'] >= valuenum) & (merged_df['ORIGINALRATE'] <= valuenum2)) |
                            ((merged_df['ORIGINALAMOUNT'] >= valuenum) & (merged_df['ORIGINALAMOUNT'] <= valuenum2))
                        )
                    ]
            else:
                if comparison_type == 'equal':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (merged_df['VALUENUM'] == valuenum)
                    ]
                elif comparison_type == 'more':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (merged_df['VALUENUM'] > valuenum)
                    ]
                elif comparison_type == 'less':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (merged_df['VALUENUM'] < valuenum)
                    ]
                elif comparison_type == 'between':
                    result = merged_df[
                        (merged_df['ITEMID'] == itemid_value_item) & 
                        (merged_df['VALUENUM'] >= valuenum) & 
                        (merged_df['VALUENUM'] <= valuenum2)
                    ]
    
    # Display results
    if len(result) == 0:
        if entity:
            print(f"This is a data frame result where the label is '{entity}': NO INFORMATION")
        else:
            print(f"This is a data frame result where the value is '{valuenum}': NO INFORMATION")
    else:
        if entity:
            print(f"This is a data frame result where the label is '{entity}'")
        else:
            print(f"This is a data frame result where the value is '{valuenum}'")
        return result