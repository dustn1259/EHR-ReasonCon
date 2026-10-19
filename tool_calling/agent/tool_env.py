from collections import OrderedDict
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
import argparse
from tabulate import tabulate  # For better table formatting in terminal
from item_search_tool import *
from data_show import *

# Item_Search lists every candidate label it matched, flagged by whether the patient
# actually has it. Only the "not present" side needs bounding — see tool_call().
ITEM_SEARCH_ABSENT_LIMIT = int(os.getenv("ITEM_SEARCH_ABSENT_LIMIT", "100"))


# EHR-ReasonCon dataset root (the PhysioNet bundle). Defaults to <release>/EHR_ReasonCon;
# override with the EHR_REASONCON_DIR env var (see check_dataset.py / README).
EHR_REASONCON_DIR = os.getenv(
    "EHR_REASONCON_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EHR_ReasonCon")),
).rstrip("/")


class TerminalSearchTool:
    def __init__(self):
        self.search_results = OrderedDict()
        self.selected_items_total = []
        self.dfs = None
        self.top_value_dict = None
        self.search_tool_params = None
        self.admitdate = None
        self.dischdate = None
        self.vertex_model = None
        
    def load_sample_data(self, hadm_id):
        """Load all data related to a patient hospital admission."""
        print(f"Loading data for HADM_ID: {hadm_id}...")
        
        # Per-patient EHR data lives under EHR_ReasonCon (download from PhysioNet)
        # MIMIC-III v1.4 CSVs (separate PhysioNet download). Default: <release>/mimiciii; override with MIMIC_DIR.
        data_storage_path = os.getenv("MIMIC_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mimiciii"))).rstrip("/")
        base_path = f'{EHR_REASONCON_DIR}/patient_data/{hadm_id}'
            
        # Define file paths for patient-specific data
        file_paths = {
            'chart_df': f'{base_path}/chartevents.csv',
            'lab_df': f'{base_path}/labevents.csv',
            'pres_df': f'{base_path}/prescriptions.csv',
            'input_mv_df': f'{base_path}/inputevents_mv.csv',
            'inputcv_df': f'{base_path}/inputevents_cv.csv',
            'micro_df': f'{base_path}/microbiologyevents.csv',
            'procedures_mv_df': f'{base_path}/procedureevents_mv.csv',
            'output_df': f'{base_path}/outputevents.csv',
            'diagnoses_icd_df': f'{base_path}/diagnosis_icd.csv',
            'procedures_icd_df': f'{base_path}/procedure_icd.csv',
        }
        
        # Define paths for reference data
        ref_paths = {
            'diagnoses_icd': f'{data_storage_path}/DIAGNOSES_ICD.csv',
            'd_icd_diagnoses': f'{data_storage_path}/D_ICD_DIAGNOSES.csv',
            'd_icd_procedures': f'{data_storage_path}/D_ICD_PROCEDURES.csv',
            'procedures_icd': f'{data_storage_path}/PROCEDURES_ICD.csv',
            # 'ditems': f'{data_storage_path}/D_ITEMS.csv',
            'ditems': f'{EHR_REASONCON_DIR}/resources/preprocessed_d_items.csv',
            'dlab': f'{data_storage_path}/D_LABITEMS.csv',
            'admissions': f'{data_storage_path}/ADMISSIONS.csv'
        }
        
        # Load all dataframes
        try:
            chart_df = pd.read_csv(file_paths['chart_df'])
            lab_df = pd.read_csv(file_paths['lab_df'])
            pres_df = pd.read_csv(file_paths['pres_df'])
            input_mv_df = pd.read_csv(file_paths['input_mv_df'])
            inputcv_df = pd.read_csv(file_paths['inputcv_df'])
            micro_df = pd.read_csv(file_paths['micro_df'])
            procedures_mv_df = pd.read_csv(file_paths['procedures_mv_df'])
            output_df = pd.read_csv(file_paths['output_df'])
            procedures_icd = pd.read_csv(file_paths['procedures_icd_df'])
            diagnoses_icd = pd.read_csv(file_paths['diagnoses_icd_df'])

            d_icd_diagnoses = pd.read_csv(ref_paths['d_icd_diagnoses'])
            d_icd_procedures = pd.read_csv(ref_paths['d_icd_procedures'])

            ditems = pd.read_csv(ref_paths['ditems'])
            dlab = pd.read_csv(ref_paths['dlab'])
            admissions = pd.read_csv(ref_paths['admissions'])
            
            # Filter admissions to only include the current hadm_id
            admissions = admissions.loc[admissions['HADM_ID'] == hadm_id, ['ADMITTIME', 'DISCHTIME']]
            admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
            admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
            admissions['admitdate'] = admissions['ADMITTIME'].dt.date
            admissions['dischdate'] = admissions['DISCHTIME'].dt.date
            
            # Create subsets of each dataframe with relevant columns
            chartevents_subset = chart_df[['ROW_ID','HADM_ID', 'ITEMID','CHARTTIME','VALUE','VALUENUM','VALUEUOM']]
            labevents_subset = lab_df[['ROW_ID','HADM_ID', 'ITEMID','CHARTTIME','VALUE','VALUENUM','VALUEUOM','FLAG']]
            prescriptions_subset = pres_df[['ROW_ID', 'HADM_ID', 'STARTDATE', 'ENDDATE', 'DRUG','PROD_STRENGTH', 'DOSE_VAL_RX',
                                          'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'ROUTE']]
            inputevents_mv_subset = input_mv_df[['ROW_ID', 'HADM_ID', 'STARTTIME', 'ENDTIME','ITEMID', 'AMOUNT', 'AMOUNTUOM', 
                                               'RATE', 'RATEUOM', 'ORDERCATEGORYNAME','SECONDARYORDERCATEGORYNAME', 
                                               'TOTALAMOUNT','TOTALAMOUNTUOM','ORIGINALAMOUNT', 'ORIGINALRATE']]
            inputevents_cv_subset = inputcv_df[['ROW_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM', 'RATE', 
                                              'RATEUOM','ORIGINALAMOUNT','ORIGINALAMOUNTUOM', 'ORIGINALROUTE', 
                                              'ORIGINALRATE', 'ORIGINALRATEUOM']]
            microbiologyevents_subset = micro_df[['ROW_ID','HADM_ID', 'CHARTDATE','SPEC_ITEMID', 'SPEC_TYPE_DESC', 
                                                'ORG_ITEMID', 'ORG_NAME','AB_NAME','INTERPRETATION']]
            procedureevents_mv_subset = procedures_mv_df[['ROW_ID','ITEMID','HADM_ID','STARTTIME','ENDTIME','LOCATION']]
            outputevents_subset = output_df[['ROW_ID','HADM_ID','CHARTTIME','ITEMID','VALUE', 'VALUEUOM']]
            diagnoses_icd_subset = diagnoses_icd[['ROW_ID','HADM_ID', 'ICD9_CODE']]
            d_icd_diagnoses_subset = d_icd_diagnoses[['ROW_ID','ICD9_CODE', 'SHORT_TITLE','LONG_TITLE']]
            d_icd_procedures_subset = d_icd_procedures[['ROW_ID','ICD9_CODE', 'SHORT_TITLE','LONG_TITLE']]
            procedures_icd_subset = procedures_icd[['ROW_ID','HADM_ID', 'ICD9_CODE']]
            d_items_subset = ditems[['ROW_ID','ITEMID', 'LABEL','LINKSTO','CATEGORY','NUMERIC_VALUE']]
            d_labitems_subset = dlab[['ROW_ID','ITEMID', 'LABEL','FLUID','CATEGORY']]
            
            # Return a dictionary of all dataframes
            return {
                'labevents': labevents_subset,
                'd_labitems': d_labitems_subset,
                'chartevents': chartevents_subset,
                'inputevents_mv': inputevents_mv_subset,
                'inputevents_cv': inputevents_cv_subset,
                'microbiologyevents': microbiologyevents_subset,
                'procedureevents_mv': procedureevents_mv_subset,
                'outputevents': outputevents_subset,
                'd_items': d_items_subset,
                'diagnoses_icd': diagnoses_icd_subset,
                'd_icd_diagnoses': d_icd_diagnoses_subset,
                'procedures_icd': procedures_icd_subset,
                'd_icd_procedures': d_icd_procedures_subset,
                'prescriptions': prescriptions_subset,
                'admissions': admissions
            }
            
        except FileNotFoundError as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def init_search(self):
        """Initialize search parameters."""
        print("Initializing search parameters...")
        
        # Define pre-dictionary for mapping search terms to table columns
        pre_dict = {
            'item (label)': {'chartevents': 'label', 'labevents': 'label', 'outputevents': 'label', 
                          'inputevents_cv': 'label','inputevents_mv': 'label','procedureevents_mv': 'label'}, 
            'value': {'labevents':'value', 'chartevents':'value','outputevents':'value'},
            'valuenum': {'labevents':'valuenum', 'chartevents':'valuenum','outputevents':'valuenum','prescriptions':'dose_val_rx'},
            'amount': {'inputevents_cv':'amount','inputevents_mv':'amount'},
            'rate': {'inputevents_cv':'rate','inputevents_mv':'rate'},
            'amountuom': {'inputevents_cv':'amountuom','inputevents_mv':'amountuom'},
            'rateuom': {'inputevents_cv':'rateuom','inputevents_mv':'rateuom'},
            'originalamount': {'inputevents_cv':'originalamount','inputevents_mv':'originalamount'},
            'originalrate': {'inputevents_cv':'originalrate','inputevents_mv':'originalrate'},
            'unit':{'chartevents':'valueuom','labevents':'valueuom','outputevents':'valueuom','prescriptions':'dose_unit_rx'},
            'charttime':{'chartevents':'charttime','labevents':'charttime','outputevents':'charttime',
                      'inputevents_cv':'charttime','microbiologyevents':'chartdate','inputevents_mv':'charttime',
                      'prescriptions':'charttime', 'procedureevents_mv': 'charttime'},
            'route':{'inputevents_cv':'originalroute','prescriptions':'route'},
            'form_unit_disp':{'prescriptions':'form_unit_disp'},
            'form_val_disp':{'prescriptions':'form_val_disp'},
            'fluid':{'labevents':'fluid'},
            'organism':{'microbiologyevents':'org_name'},
            'specimen':{'microbiologyevents':'spec_type_desc'},
            'interpretation':{'microbiologyevents':'interpretation'},
            'ab_name':{'microbiologyevents':'ab_name'},
            'location': {'procedureevents_mv': 'location'},
            'flag': {'labevents': 'flag'}
        }
        
        # Load the last row ID and hadm_id
        try:            
            note_test_valid_hadmid = load_last_row_id()
            note_test_valid_hadmid = note_test_valid_hadmid.split(', ')
            NOTE, TEST_VAL, current_hadm = note_test_valid_hadmid[0], note_test_valid_hadmid[1], float(note_test_valid_hadmid[2])
            
            note_file, target_list, note_file_path = get_target_list(TEST_VAL, NOTE)
            
            row_ids = note_file.index[note_file['HADM_ID'] == float(current_hadm)].tolist()
            last_row_id = row_ids[0]
            
            lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, drug_list, sets = read_data()
            
            rows = read_csv(note_file_path, last_row_id)
            row_iterator = iter(rows)
            
            try:
                index, row_id, hadm_id, admission_str, charttime_str = next(row_iterator)
                return pre_dict, lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, drug_list, sets, index, row_id, hadm_id, admission_str, charttime_str
            except StopIteration:
                print("No more rows to process.")
                return pre_dict, None, None, None, None, None, None, None, None, None, None, None, None
                
        except ImportError:
            print("Warning: Could not load item_search_tool functions. Using default hadm_id 100001.")
            # Return defaults if imports fail
            return pre_dict, None, None, None, None, None, None, None, None, None, 100001, None, None
    
    def get_tables(self, table1):
        """Get the data tables based on the table name."""
        if self.dfs is None:
            print("Error: Data not loaded.")
            return None, None
        
        if table1 == 'labevents':
            return self.dfs['labevents'], self.dfs['d_labitems']
        elif table1 == 'diagnoses_icd':
            return self.dfs['diagnoses_icd'], self.dfs['d_icd_diagnoses']
        elif table1 == 'procedures_icd':
            return self.dfs['procedures_icd'], self.dfs['d_icd_procedures']
        elif table1 == 'prescriptions':
            return self.dfs['prescriptions'], None
        else:
            return self.dfs[table1], self.dfs['d_items']
        
    def find_all_items_from_dicts(self, category):
        """Find all items from a specific category."""
        print(f"Finding all items in category '{category}'...")
        
        # Basic check to ensure dfs is loaded
        if self.dfs is None:
            print("Error: Data not loaded.")
            return pd.DataFrame()
        
        try:
            if category == 'lab':
                # Get all unique lab items
                lab_items = self.dfs['labevents']['ITEMID'].unique()
                result_df = self.dfs['d_labitems'][self.dfs['d_labitems']['ITEMID'].isin(lab_items)]
                return result_df
            elif category == 'pres':
                # Get all unique prescription drugs
                result_df = self.dfs['prescriptions'][['DRUG', 'ROUTE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX']].drop_duplicates()
                return result_df
            elif category == 'd_items':
                # Get all d_items that appear in chartevents
                chart_items = self.dfs['chartevents']['ITEMID'].unique()
                result_df = self.dfs['d_items'][self.dfs['d_items']['ITEMID'].isin(chart_items)]
                return result_df
            else:
                print(f"Invalid category: {category}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error in find_all_items_from_dicts: {str(e)}")
            return pd.DataFrame()
    
    def find(self, word):
        """Search for data based on input parameters."""
        print(f"Searching for '{word}'...")
        try:
            all_items = search_tool(word, *self.search_tool_params)
            return {
                'type': 'list',
                'data': all_items
            }
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Search error: {str(e)}",
                'data': [[f"{word}_error", f"Error occurred"]]
            }
    
    def get_values(self):
        val1 = input("Value1 (required): ").strip()
        cmp_opts = ['equal', 'more', 'less', 'between']
        cmp_type = input(f"Comparison ({'/'.join(cmp_opts)}): ").strip().lower()
        if cmp_type not in cmp_opts:
            print(f"[Error] Invalid comparison type: {cmp_type}")
            return "", ""
        val2 = None
        if cmp_type == 'between':
            val2 = input("Value2 (for between): ").strip()
        return cmp_type, val1, val2
    
    def get_times(self):
        print("Time expression options:")
        print("  1: Time1")
        print("  2: (Time1 -1day) ~ (Time1 +1day)")
        print("  3: (Time1 -1day) ~ (Time2 +1day)")
        texp = input("Choose 1/2/3: ").strip()
        if texp not in ('1', '2', '3'):
            print(f"[Error] Invalid time expression: {texp}")
            return "", "", ""

        # 4) Enter Time1
        t1_str = input("Time1 (YYYY-MM-DD): ").strip()
        include_time = False
        if texp == '1':
            inc = input("Include clock time? (y/N): ").strip().lower()
            if inc == 'y':
                include_time = True
                t1_clock = input("Time1 clock (HH:MM:SS): ").strip()
                t1_full = f"{t1_str} {t1_clock}"
                try:
                    date1 = datetime.strptime(t1_full, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print("[Error] Invalid datetime format")
                    return "", "", ""
            else:
                try:
                    date1 = datetime.strptime(t1_str, '%Y-%m-%d').date()
                except ValueError:
                    print("[Error] Invalid date format")
                    return "", "", ""
        else:
            try:
                date1 = datetime.strptime(t1_str, '%Y-%m-%d').date()
            except ValueError:
                print("[Error] Invalid date format")
                return "", "", ""

        # 5) Enter Time2 (option 3)
        date2 = None
        if texp == '3':
            t2_str = input("Time2 (YYYY-MM-DD): ").strip()
            try:
                date2 = datetime.strptime(t2_str, '%Y-%m-%d').date()
            except ValueError:
                print("[Error] Invalid date format")
                return "", "", ""

        return texp, date1, date2

    def tool_init(self, vertex_model):
        """Run the search tool in interactive mode."""
        print("="*80)
        print("MEDICAL DATA SEARCH TOOL - INTERACTIVE MODE")
        print("="*80)
        print("Type 'exit' to quit.")
        print("Loading data and initializing search parameters...")
        
        # Initialize search parameters
        self.search_tool_params = self.init_search()
        hadm_id = int(self.search_tool_params[10]) if self.search_tool_params[10] is not None else 100001
        
        # Load data
        self.dfs = self.load_sample_data(hadm_id)
        if self.dfs is None:
            print("Failed to load data. Exiting.")
            return
        # Try to load top values dictionary
        try:
            with open(f'{EHR_REASONCON_DIR}/resources/top_values.json', 'r', encoding='utf-8') as f:
                self.top_value_dict = json.load(f)
        except:
            print("Could not load top_values.json file. Some functionality may be limited.")
        
        # Set admit and discharge dates
        try:
            self.admitdate = list(self.dfs['admissions']['admitdate'])[0]
            self.dischdate = list(self.dfs['admissions']['dischdate'])[0]
            print(f"Patient admitted on {self.admitdate} and discharged on {self.dischdate}")
        except:
            print("Could not determine admission and discharge dates.")
        self.vertex_model = vertex_model

    def time_range_proc(self, time_range):
        texp, date1, date2 = "", "", ""
        if ' to ' in time_range:
            texp = "4"
            date1 = time_range.split(' to ')[0]
            date2 = time_range.split(' to ')[1]
        elif 'between' in time_range and 'and' in time_range:
            texp = "4"
            time_range = time_range.replace('between', '')
            date1 = time_range.split('and')[0].strip()
            date2 = time_range.split('and')[1].strip()
        else:
            texp = '1'
            date1 = time_range
            date2 = None
        return texp, date1, date2
    
    def value_range_proc(self, value_range):
        cmp_type, val1, val2 = "", None, None
        if value_range.count('<') == 2:
            cmp_type = "between"
            val1 = value_range.split('<')[0]
            val2 = value_range.split('<')[-1]
        elif value_range.count('<') == 1:
            cmp_type = "less"
            val1 = value_range.split('<')[-1]
        elif '>' in value_range:
            cmp_type = "more"
            val1 = value_range.split('>')[-1]
        elif '=' in value_range:
            cmp_type = "equal"
            val1 = value_range.split('=')[-1]
        else:
            cmp_type = "equal"
            val1 = value_range.strip()
        return cmp_type, val1, val2
    
    def category_by_table(self):
        tables = list(self.dfs.keys())
        all_tabs = ""
        categories = " ("
        for table in tables:
            if 'd_' in table or table in ['microbiologyevents', 'prescriptions', 'admissions']:
                continue
            df_main, df_dict = self.get_tables(table)
            if (df_dict is not None) and ('CATEGORY' in df_dict.columns):
                merged = pd.merge(df_main, df_dict, on='ITEMID', how='left')
                categories_ = sorted(merged['CATEGORY'].dropna().unique().tolist())   
                categories += f"{table}: {categories_}. "
                all_tabs += table + ", "
        categories += ')'
        return all_tabs[:-2], categories

    def tool_call(self, search_params):
        # Main interactive loop
        tool = search_params['tool']
        params = search_params['params']
        subnote = search_params['subnote']

        if tool == 'Item_Search':
            word = params["The entity to search"]
            result = self.find(word)
            data_list = result['data']
            final_result = {}
            for d_point in data_list:
                if 'Error occurred' in d_point:
                    continue
                try:
                    final_result[d_point[1]][d_point[2]].append(d_point[0])
                except:
                    try:
                        final_result[d_point[1]][d_point[2]] = [d_point[0]]
                    except:
                        final_result[d_point[1]] = {d_point[2]: [d_point[0]]}
            
            string_res = ""
            
            for tab, tab_data in final_result.items():
                string_res += f"\n\n{tab}"
                for exist, list_ in tab_data.items():
                    if exist == 1:
                        string_res += f"\nITEMs (exists in the patient records): {list_}"
                    else:
                        # The n-gram candidate generator is very permissive on ICD titles
                        # (a query like "Sodium" yields >10k non-present candidates, i.e. a
                        # 500KB tool result). Present-in-record items are what the caller
                        # needs, so only a sample of the rest is shown — and the count that
                        # was dropped is stated, never silently truncated.
                        shown = list_[:ITEM_SEARCH_ABSENT_LIMIT]
                        string_res += f"\nITEMs (does not exist in the patient records): {shown}"
                        if len(list_) > len(shown):
                            string_res += f" ... and {len(list_) - len(shown)} more not shown"

            return string_res

        elif tool == 'Semantic_Search':
            table_name = params["The table to check"]
            sed_entity = params["The entity to search"]
            
            df = find_items_from_dicts(subnote, table_name, sed_entity, self.dfs, self.vertex_model)
            try:
                return df.reset_index().drop(columns=['ROW_ID', 'ITEMID', 'index'])
            except:
                return df.reset_index().drop(columns=['ROW_ID', 'index'])


        elif tool == 'Top_Values_for_Entities':
            table_name = params["The table to check"]
            top_entity = params["The item to check"]
            result_list = top_values(top_entity, table_name, self.top_value_dict)
            return result_list
        
        elif tool == 'Table_Value_Time':
            table = params["The table to check"]
            value_range = params["Value Range"]
            time_range = params["Time Range"]
            
            cmp_type, val1, val2 = self.value_range_proc(value_range)
            texp, date1, date2 = self.time_range_proc(time_range)

            # 6) Load the table and reference table
            df_main, df_ref = self.get_tables(table)

            # 7) Run the actual search
            try:
                if 'icd' in table.lower():
                    return df_main
                elif 'micro' in table.lower() or 'procedure' in table.lower():
                    result_df = search_table(
                        df_main, df_ref, table, 
                        entity=None, 
                        search_type='date_only', 
                        valuenum=None, 
                        valuenum2=None, 
                        comparison_type='equal',  # ignored
                        date1=date1, 
                        date2=date2, 
                        time_expression=texp, 
                        category=None
                    )
                else:
                    result_df = search_table(
                        df_main, df_ref, table,
                        entity=None,
                        search_type='date_value',
                        valuenum=val1,
                        valuenum2=val2,
                        comparison_type=cmp_type,
                        date1=date1,
                        date2=date2,
                        time_expression=texp,
                        category=None
                    )

                if result_df is not None:
                    sort_keys = ['CHARTDATE', 'CHARTTIME', 'STARTDATE', 'STARTTIME']
                    sort_key = next((key for key in sort_keys if key in result_df.columns), None)
                    if sort_key:
                        result_df.sort_values(by=sort_key, ascending=True, inplace=True)
                try:
                    return result_df.reset_index().drop(columns=['ROW_ID', 'index'])
                except:
                    try:
                        return result_df.reset_index().drop(columns=['ROW_ID_x', 'ROW_ID_y', 'HADM_ID', 'index', 'ITEMID'])
                    except:
                        return result_df

            except Exception as e:
                print(f"[Error] TVT search failed: {e}")
            
        elif tool == 'Table_Category_Time':
            # Table-Category-Time search function
            # 1) Select the table
            table_name = params["The table to check"]
            time_range = params["Time Range"]
            category = params["Category"]
            
            # 3) Select the time expression
            texp, date1, date2 = self.time_range_proc(time_range)
            df_main, df_ref = self.get_tables(table_name)

            # 5) Call search_table
            try:
                result_df = search_table(
                    df_main, df_ref, table_name,
                    entity=None, search_type='category_date',
                    valuenum=None, valuenum2=None,
                    comparison_type=None,
                    date1=date1, date2=date2,
                    time_expression=texp, category=category
                )
                try:
                    return result_df.reset_index().drop(columns=['ROW_ID', 'index'])
                except:
                    return result_df
            except Exception as e:
                print(f"Error during TCT search: {e}")
        
        elif tool == 'Table_Selected_Item_Time':
            # 1) Select the time option
            time_range = params["Time Range"]
            word = params["The item to check"]
            table = params["The table to check"]
            texp, date1, date2 = self.time_range_proc(time_range)
            
            self.selected_items_total = [(word, table)]
                        
            df_main, df_ref = self.get_tables(table)
            if 'icd' in table.lower():
                return df_main
            else:
                result_df = search_table(
                    df_main, 
                    df_ref, 
                    table, 
                    entity=word, 
                    search_type='entity_date', 
                    comparison_type=None,
                    valuenum=None,  
                    valuenum2=None, 
                    date1=date1, 
                    date2=date2,
                    time_expression=texp,  
                    category=None
                )
            if result_df is not None:
                sort_keys = ['CHARTDATE', 'CHARTTIME', 'STARTDATE', 'STARTTIME']
                sort_key = next((key for key in sort_keys if key in result_df.columns), None)
                if sort_key:
                    result_df.sort_values(by=sort_key, ascending=True, inplace=True)
            try:
                return result_df.reset_index().drop(columns=['ROW_ID', 'index'])
            except:
                try:
                    return result_df.reset_index().drop(columns=['ROW_ID_x', 'ROW_ID_y', 'HADM_ID', 'index', 'ITEMID'])
                except:
                    return result_df
        elif tool == 'Table_Selected_Item_Value_Time':
            # 1) Select the table
            table = params["The table to check"]
            time_range = params["Time Range"]
            value_range = params["Value Range"]
            word = params["The item to check"]
            self.selected_items_total = [(word, table)]
            
            cmp_type, val1, val2 = self.value_range_proc(value_range)
            texp, date1, date2 = self.time_range_proc(time_range)
            df1, df2 = self.get_tables(table)
            if 'icd' in table.lower():
                return df1
            elif table == 'microbiologyevents' or table == 'procedureevents_mv':
                result_df = search_table(
                    df1, 
                    df2, 
                    table, 
                    entity=word, 
                    search_type='entity_date', 
                    comparison_type=None,
                    valuenum=None,  
                    valuenum2=None, 
                    date1=date1, 
                    date2=date2,
                    time_expression=texp,  
                    category=None
                )
            else:
                result_df = search_table(
                    df1, 
                    df2, 
                    table, 
                    entity=word, 
                    search_type='entity_value_date', 
                    comparison_type=cmp_type,
                    valuenum=val1,  # adjust as needed
                    valuenum2=val2, 
                    date1=date1, 
                    date2=date2,
                    time_expression=texp,  # adjust as needed
                    category=None
                )
            if result_df is not None:
                sort_keys = ['CHARTDATE', 'CHARTTIME', 'STARTDATE', 'STARTTIME']
                sort_key = next((key for key in sort_keys if key in result_df.columns), None)
                if sort_key:
                    result_df.sort_values(by=sort_key, ascending=True, inplace=True)
            
            try:
                return result_df.reset_index().drop(columns=['ROW_ID', 'index'])
            except:
                try:
                    return result_df.reset_index().drop(columns=['ROW_ID_x', 'ROW_ID_y', 'HADM_ID', 'index', 'ITEMID'])
                except:
                    return result_df
        elif tool == 'Table_Time':
            # 1) Select the table
            table = params["The table to check"]
            time_range = params["Time Range"]
            texp, date1, date2 = self.time_range_proc(time_range)

            # 6) Load the table and reference table
            df_main, df_ref = self.get_tables(table)

            # 7) Run the actual search
            try:
                if 'icd' in table.lower():
                    return df_main
                else:
                    result_df = search_table(
                        df_main, df_ref, table, entity=None, search_type='date_only', 
                        valuenum=None, valuenum2=None, 
                        comparison_type='equal',  # ignored
                        date1=date1, date2=date2, 
                        time_expression=texp, category=None
                    )
                if result_df is not None:
                    sort_keys = ['CHARTDATE', 'CHARTTIME', 'STARTDATE', 'STARTTIME']
                    sort_key = next((key for key in sort_keys if key in result_df.columns), None)
                    if sort_key:
                        result_df.sort_values(by=sort_key, ascending=True, inplace=True)
                try:
                    return result_df.reset_index().drop(columns=['ROW_ID', 'index'])
                except:
                    try:
                        return result_df.reset_index().drop(columns=['ROW_ID_x', 'ROW_ID_y', 'HADM_ID', 'index', 'ITEMID'])
                    except:
                        return result_df
            except Exception as e:
                print(f"[Error] TT search failed: {e}")        