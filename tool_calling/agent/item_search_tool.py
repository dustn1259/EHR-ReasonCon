import sqlite3
import json
import pandas as pd
from tqdm import tqdm
import re
import time
import textdistance
from strsimpy.metric_lcs import MetricLCS
from strsimpy.cosine import Cosine
from strsimpy.ngram import NGram
from datetime import datetime, timedelta
import re
import sqlite3
from itertools import combinations
import copy
import sqlite3
import os
import pandas as pd
from itertools import product
import warnings
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# EHR-ReasonCon dataset root (the PhysioNet bundle). Defaults to <release>/EHR_ReasonCon;
# override with the EHR_REASONCON_DIR env var (see check_dataset.py / README).
EHR_REASONCON_DIR = os.getenv(
    "EHR_REASONCON_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EHR_ReasonCon")),
).rstrip("/")


def read_data():
    print("Reading data...")
    
    sets = pd.read_csv('../final_set_of_abbreviation1.csv')
    abb_df = pd.DataFrame(sets, columns=['abbreviation', 'expansion', 'whole_set'])

    data_path = os.getenv("MIMIC_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mimiciii"))).rstrip("/") + "/"
    
    lab_id = pd.read_csv(f'{data_path}D_LABITEMS.csv')
    d_items = pd.read_csv(f'{data_path}D_ITEMS.csv')
    prescriptions = pd.read_csv(f'{data_path}PRESCRIPTIONS.csv')
    
    drug_list = list(set(list(prescriptions['DRUG'])))

    d_icd_diagnoses = pd.read_csv(f'{data_path}D_ICD_DIAGNOSES.csv')
    d_icd_procedures = pd.read_csv(f'{data_path}D_ICD_PROCEDURES.csv')

    return lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, drug_list, sets


def get_target_list(task_val, note):
    if task_val == 'val':
        task_val = 'valid'
    path_ = f'{EHR_REASONCON_DIR}/{note}/{task_val}'
    note_file_path = f'{path_}/{note}_'
    try:
        note_file = pd.read_csv(note_file_path + 'test.csv')
        note_file_path = note_file_path + 'test.csv'
    except:
        note_file = pd.read_csv(note_file_path + 'val.csv')
        note_file_path = note_file_path + 'val.csv'
    
    hadm_list = list(note_file['HADM_ID'])
    return note_file, hadm_list, note_file_path


def create_dicts(lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures):
    """ Make label to id dictionary {label: item id}
    
    Returns:
        lower_label_itemid_dict: lab label to item id
        d_item_lower_label_itemid_dict: d_item label to item id
        prescriptions_itemid_dict: drug to id
    """
    lower_label_itemid_dict = {}
    d_item_lower_label_itemid_dict = {}
    prescriptions_itemid_dict = {}
    drug_list = list(set(list(prescriptions['DRUG'])))

    d_icd_diag_lower_label_itemid_dict = {}
    d_icd_proc_lower_label_itemid_dict = {}

    for i in range(len(lab_id)):
        itemid = lab_id['ITEMID'][i]
        try:
            lower_label_itemid_dict[lab_id['LABEL'][i].lower()].append(itemid)
        except:
            lower_label_itemid_dict[lab_id['LABEL'][i].lower()] = [itemid]

    for i in range(len(d_items)):
        itemid = d_items['ITEMID'][i]
        try:
            d_item_lower_label_itemid_dict[d_items['LABEL'][i].lower()].append(itemid)
        except:
            try:
                d_item_lower_label_itemid_dict[d_items['LABEL'][i].lower()] = [itemid]
            except:
                pass

    for i in range(len(drug_list)):
        itemid = i
        try:
            prescriptions_itemid_dict[drug_list[i].lower()].append(itemid)
        except:
            try:
                prescriptions_itemid_dict[drug_list[i].lower()] = [itemid]
            except:
                pass


    for i in range(len(d_icd_diagnoses)):
        icd_code = d_icd_diagnoses['ICD9_CODE'][i]
        try:
            d_icd_diag_lower_label_itemid_dict[d_icd_diagnoses['SHORT_TITLE'][i].lower()].append(icd_code)
        except:
            d_icd_diag_lower_label_itemid_dict[d_icd_diagnoses['SHORT_TITLE'][i].lower()] = [icd_code]

        try:
            d_icd_diag_lower_label_itemid_dict[d_icd_diagnoses['LONG_TITLE'][i].lower()].append(icd_code)
        except:
            d_icd_diag_lower_label_itemid_dict[d_icd_diagnoses['LONG_TITLE'][i].lower()] = [icd_code]


    for i in range(len(d_icd_procedures)):
        icd_code = d_icd_procedures['ICD9_CODE'][i]
        try:
            d_icd_proc_lower_label_itemid_dict[d_icd_procedures['SHORT_TITLE'][i].lower()].append(icd_code)
        except:
            d_icd_proc_lower_label_itemid_dict[d_icd_procedures['SHORT_TITLE'][i].lower()] = [icd_code]

        try:
            d_icd_proc_lower_label_itemid_dict[d_icd_procedures['LONG_TITLE'][i].lower()].append(icd_code)
        except:
            d_icd_proc_lower_label_itemid_dict[d_icd_procedures['LONG_TITLE'][i].lower()] = [icd_code]

    label_keys = list(lower_label_itemid_dict.keys())
    d_item_label_keys = list(d_item_lower_label_itemid_dict.keys())
    prescription_keys = list(prescriptions_itemid_dict.keys())
    d_icd_diag_keys = list(d_icd_diag_lower_label_itemid_dict.keys())
    d_icd_proc_keys = list(d_icd_proc_lower_label_itemid_dict.keys())

    return lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, drug_list, d_icd_diag_lower_label_itemid_dict, d_icd_diag_keys, d_icd_proc_lower_label_itemid_dict, d_icd_proc_keys


def abbreviation_checker(A, B):
    index = 0
    B = re.sub(r'\(.*?\)', '', B)
    for char in A:
        if char == B[index]:
            index += 1
            if index == len(B):
                return True
    return False

def bigram_search_result_lab(sets, lab_id, target, label_keys, lower_label_itemid_dict, n=2, fir_thr=0.5, sec_thr=0.7):
    """
    Results of N-gram search in lab items
    """
    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list

    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in label_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += lower_label_itemid_dict[label]
    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)
    
    return final_item_id_candidates, lab_id[lab_id['ITEMID'].isin(final_item_id_candidates)]

def bigram_search_result_d_item(sets, d_items, target, d_item_label_keys, d_item_lower_label_itemid_dict, n=2, fir_thr=0.5, sec_thr=0.7):
    """
    Results of N-gram search in d_items
    """

    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list

    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += d_item_lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in d_item_label_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)

    return final_item_id_candidates, d_items[d_items['ITEMID'].isin(final_item_id_candidates)]


def bigram_search_result_d_icd_item(sets, d_icd_diagnoses, target, d_icd_diag_keys, d_icd_diag_lower_label_itemid_dict, n=2, fir_thr=0.5, sec_thr=0.7):
    """
    Results of N-gram search in d_items
    """

    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list

    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += d_icd_diag_lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in d_icd_diag_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += d_icd_diag_lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += d_icd_diag_lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += d_icd_diag_lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += d_icd_diag_lower_label_itemid_dict[label]
    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)

    return final_item_id_candidates, d_icd_diagnoses[d_icd_diagnoses['ICD9_CODE'].isin(final_item_id_candidates)]



def bigram_search_result_prescription(sets, prescriptions, drug_list, target, d_item_label_keys, d_item_lower_label_itemid_dict, n=2, fir_thr=0.5, sec_thr=0.7):
    """
    Results of N-gram search in prescriptions
    """
    
    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list
    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += d_item_lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in d_item_label_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]

    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)

    drug_answers = [drug_list[k] for k in final_item_id_candidates]

    return drug_answers


def filter_entity(original_entity, target_entity):
    if original_entity.lower() == 'rr' or 'rr ' in original_entity.lower() or 'respiratory rate' in original_entity.lower():
        if target_entity.lower() == 'rr' or 'rr ' in target_entity.lower() or 'respiratory rate' in target_entity.lower():
            return True
        else:
            False
    elif 'calcium' in original_entity.lower() or 'ca+' in original_entity.lower():
        if 'ca' in target_entity.lower():
            return True
        else:
            False
    else:
        return True

def total_checker(lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, predef_map, entity_name, hadm_id, admission_str, charttime_str, conn, cursor, sets, lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, drug_list, d_icd_diag_lower_label_itemid_dict, d_icd_diag_keys, d_icd_proc_lower_label_itemid_dict, d_icd_proc_keys, n=2, fir_thr=0.5, sec_thr=0.7):
    tag_value = "mimic"
    final_answer = []
    all_answer = []
    
    search_entities_list = [entity_name.lower()]
    try:
        predef_mappings = predef_map[entity_name.lower()]
        search_entities_list.extend(predef_mappings)
    except:
        pass
    
    for predef_ent in search_entities_list:
        lab_results = bigram_search_result_lab(sets, lab_id, predef_ent, label_keys, lower_label_itemid_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
        try:
            from_d_lab_items = pd.concat([from_d_lab_items, lab_results[1]])
            from_d_lab_items = from_d_lab_items.drop_duplicates(subset='ITEMID')
        except:
            from_d_lab_items = lab_results[1]

        d_item_results = bigram_search_result_d_item(sets, d_items, predef_ent, d_item_label_keys, d_item_lower_label_itemid_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
        try:
            from_d_items = pd.concat([from_d_items, d_item_results[1]])
            from_d_items = from_d_items.drop_duplicates(subset='ITEMID')
        except:
            from_d_items = d_item_results[1]
        
        prescription_results = list(bigram_search_result_prescription(sets, prescriptions, drug_list, predef_ent, prescription_keys, prescriptions_itemid_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr))
        try:
            from_prescription.extend(prescription_results)
        except:
            from_prescription = prescription_results
        
        d_icd_diag_results = bigram_search_result_d_icd_item(sets, d_icd_diagnoses, predef_ent, d_icd_diag_keys, d_icd_diag_lower_label_itemid_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
        try:
            from_d_icd_diag = pd.concat([from_d_icd_diag, d_icd_diag_results[1]])
            from_d_icd_diag = from_d_icd_diag.drop_duplicates(subset='ICD9_CODE')
        except:
            from_d_icd_diag = d_icd_diag_results[1]

        d_icd_proc_results = bigram_search_result_d_icd_item(sets, d_icd_procedures, predef_ent, d_icd_proc_keys, d_icd_proc_lower_label_itemid_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
        try:
            from_d_icd_proc = pd.concat([from_d_icd_proc, d_icd_proc_results[1]])
            from_d_icd_proc = from_d_icd_proc.drop_duplicates(subset='ICD9_CODE')
        except:
            from_d_icd_proc = d_icd_proc_results[1]


    title_list=list(from_d_icd_diag['SHORT_TITLE']) + list(from_d_icd_diag['LONG_TITLE'])
    for _, title_entity in enumerate(title_list):
        if not filter_entity(entity_name, title_entity):
            continue
        title_entity = title_entity.replace("'", "''")
        sql_query = f"SELECT COUNT(*) FROM diagnoses di JOIN d_icd_diagnoses d ON di.icd9_code = d.icd9_code WHERE di.hadm_id = {hadm_id} AND (d.short_title IN ('{title_entity.lower()}','{title_entity}') OR d.long_title IN ('{title_entity.lower()}','{title_entity}'))"
        cursor.execute(sql_query)
        results = cursor.fetchall()
        all_answer.append([title_entity,'diagnoses_icd'])
        if results[0][0] > 0:     
            if [title_entity, 'diagnoses_icd'] not in final_answer:   
                final_answer.append([title_entity,'diagnoses_icd'])
                continue

    title_list=list(from_d_icd_proc['SHORT_TITLE']) + list(from_d_icd_proc['LONG_TITLE'])
    for _, title_entity in enumerate(title_list):
        if not filter_entity(entity_name, title_entity):
            continue
        title_entity = title_entity.replace("'", "''")
        sql_query = f"SELECT COUNT(*) FROM procedures di JOIN d_icd_procedures d ON di.icd9_code = d.icd9_code WHERE di.hadm_id = {hadm_id} AND (d.short_title IN ('{title_entity.lower()}','{title_entity}') OR d.long_title IN ('{title_entity.lower()}','{title_entity}'))"
        cursor.execute(sql_query)
        results = cursor.fetchall()
        all_answer.append([title_entity,'procedures_icd'])
        if results[0][0] > 0:     
            if [title_entity, 'procedures_icd'] not in final_answer:   
                final_answer.append([title_entity,'procedures_icd'])
                continue
    
    itemid=from_d_lab_items['LABEL']
    itemid_list=list(itemid)
    for idx,lab_entity in enumerate(itemid_list):
        if not filter_entity(entity_name, lab_entity):
            continue
        sql_query = f"SELECT COUNT(*) FROM labevents AS l JOIN d_labitems AS d ON l.itemid = d.itemid WHERE l.hadm_id = {hadm_id} AND date(l.charttime) BETWEEN date('{admission_str}', '-1 day') AND date('{charttime_str}', '+1 day') AND d.label = '{lab_entity.lower()}'"
        cursor.execute(sql_query)
        results = cursor.fetchall()
        all_answer.append([lab_entity,'labevents'])
        if results[0][0] > 0:     
            if [lab_entity, 'labevents'] not in final_answer:   
                final_answer.append([lab_entity,'labevents'])
                continue

    from_d_items = from_d_items[from_d_items['LINKSTO'].isin(["chartevents", "inputevents_cv", "outputevents", "inputevents_mv", "microbiologyevents", "procedureevents_mv"])]
    d_items_top_ids = []
    from_d_items_to_print = from_d_items[['ITEMID', 'LABEL', 'LINKSTO']]
    from_d_items_list = from_d_items_to_print.values.tolist()

    ditemid=from_d_items['LABEL']
    dtable=from_d_items['LINKSTO']
    ditemid_list=list(ditemid)
    dtable_list=list(dtable)
    for idx2,entity in enumerate(ditemid_list):
        if not filter_entity(entity_name, entity):
            continue
        table_name = dtable_list[idx2]
        all_answer.append([entity,table_name])
        if table_name == 'microbiologyevents':
            sql_query = f"SELECT COUNT(*) FROM microbiologyevents AS t JOIN d_items AS d ON (d.itemid = t.spec_itemid OR d.itemid = t.org_itemid) WHERE t.hadm_id = {hadm_id} AND date(t.chartdate) BETWEEN date('{admission_str}', '-1 day') AND date('{charttime_str}', '+1 day') AND (t.spec_type_desc = '{entity.lower()}' OR t.org_name = '{entity.lower()}')"
            cursor.execute(sql_query)
            results = cursor.fetchall()
            if results[0][0] > 0:    
                if [entity,table_name] not in final_answer:     
                    final_answer.append([entity,table_name])
                    continue
        elif table_name == 'inputevents_mv' or table_name == 'procedureevents_mv':
            sql_query = f"SELECT COUNT(*) FROM {table_name} AS t JOIN d_items AS d ON t.itemid = d.itemid WHERE t.hadm_id = {hadm_id} AND d.label = '{entity.lower()}'"
            cursor.execute(sql_query)
            results = cursor.fetchall()
            if results[0][0] > 0:    
                if [entity,table_name] not in final_answer:     
                    final_answer.append([entity,table_name])
                    continue
        else:
            try:
                sql_query = f"SELECT COUNT(*) FROM {table_name} AS t JOIN d_items AS d ON t.itemid = d.itemid WHERE t.hadm_id = {hadm_id} AND date(t.charttime) BETWEEN date('{admission_str}', '-1 day') AND date('{charttime_str}', '+1 day') AND d.label = '{entity.lower()}'"
                cursor.execute(sql_query)
                results = cursor.fetchall()
                if results[0][0] > 0:    
                    if [entity,table_name] not in final_answer:     
                        final_answer.append([entity,table_name])
                        continue
            except:
                continue
    for idx3,drug in enumerate(from_prescription):
        sql_query = f"SELECT COUNT(*) FROM prescriptions AS p WHERE p.hadm_id = {hadm_id} and p.drug = '{drug.lower()}'"
        cursor.execute(sql_query)
        results = cursor.fetchall()
        all_answer.append([drug,'prescriptions'])
        if results[0][0] > 0:     
            if [drug,'prescriptions'] not in final_answer:    
                final_answer.append([drug,'prescriptions'])

    new_data_answer = []

    for ans in final_answer:
        new_data_answer.append([ans[0], ans[1], 1])
    
    for ans_ in all_answer:
        if [ans_[0], ans_[1], 1] not in new_data_answer:
            new_data_answer.append([ans_[0], ans_[1], 0])

    return new_data_answer

     

def read_csv(file_name, start_row):
    df = pd.read_csv(file_name)
    for index, row in df.iterrows():
        if index < start_row: 
            continue
        row_id = row['ROW_ID']
        hadm_id = row['HADM_ID']
        admission_time = row['ADMITTIME']
        admission_time = datetime.strptime(admission_time, '%Y-%m-%d %H:%M:%S')
        admission_str = admission_time.strftime('%Y-%m-%d')
        charttime_str = row['CHARTDATE']
        yield index,row_id, hadm_id, admission_str, charttime_str


def load_last_row_id():
    try:
        with open(f'../current_hadm_id.txt', 'r') as file:
            note_task_hadm_id = file.read()
        return note_task_hadm_id
    except FileNotFoundError:
        return "discharge, val, -1" 

def ensure_patient_db(db_path, hadm_id, lab_id, d_items, d_icd_diagnoses, d_icd_procedures):
    """Build the per-patient SQLite DB from EHR_ReasonCon/patient_data if it is missing.

    Mirrors make_specific_db from the original data-construction script, but sources the
    per-admission event tables from the PhysioNet bundle instead of the raw MIMIC dumps.
    """
    if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
        return
    hadm_dir = f'{EHR_REASONCON_DIR}/patient_data/{int(float(hadm_id))}'
    event_tables = {'chartevents.csv': 'chartevents', 'prescriptions.csv': 'prescriptions',
                    'inputevents_cv.csv': 'inputevents_cv', 'inputevents_mv.csv': 'inputevents_mv',
                    'microbiologyevents.csv': 'microbiologyevents', 'outputevents.csv': 'outputevents',
                    'labevents.csv': 'labevents', 'procedureevents_mv.csv': 'procedureevents_mv'}
    icd_tables = {'diagnosis_icd.csv': 'diagnoses', 'procedure_icd.csv': 'procedures'}
    icd_cols = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']
    print(f"Building patient database {db_path} ...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for csv_name, table in event_tables.items():
            pd.read_csv(f'{hadm_dir}/{csv_name}', low_memory=False).to_sql(table, conn, if_exists='replace', index=False)
        for csv_name, table in icd_tables.items():
            # ICD9_CODE must stay TEXT: leading-zero codes (e.g. '0412' vs '412') are
            # distinct, and SQLite's numeric affinity would conflate them if the
            # column were stored as INTEGER
            df = pd.read_csv(f'{hadm_dir}/{csv_name}', low_memory=False, dtype={'ICD9_CODE': str})
            # shipped CSVs are pre-joined with ICD titles; an ambiguous numeric ICD9
            # code can match two titles, duplicating the raw row
            df[[c for c in icd_cols if c in df.columns]].drop_duplicates().to_sql(table, conn, if_exists='replace', index=False)
        lab_id.to_sql('d_labitems', conn, if_exists='replace', index=False)
        d_items.to_sql('d_items', conn, if_exists='replace', index=False)
        d_icd_diagnoses.to_sql('d_icd_diagnoses', conn, if_exists='replace', index=False)
        d_icd_procedures.to_sql('d_icd_procedures', conn, if_exists='replace', index=False)
        conn.close()
    except Exception:
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)  # never leave a half-built DB behind
        raise


def process_data(entity, pre_dict, lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, drug_list, sets, index, row_id, hadm_id, admission_str, charttime_str):
    session_data = {}
    print('HADM_ID:', hadm_id, ' Admission date:', admission_str, ' Chart date:', charttime_str)

    db_path = f'../new_final_db/sample_{hadm_id}_a_p.db'
    # Create the patient-specific database on first use (restores the original
    # make_specific_db behaviour, sourced from EHR_ReasonCon/patient_data)
    ensure_patient_db(db_path, hadm_id, lab_id, d_items, d_icd_diagnoses, d_icd_procedures)
    lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, drug_list, d_icd_diag_lower_label_itemid_dict, d_icd_diag_keys, d_icd_proc_lower_label_itemid_dict, d_icd_proc_keys = create_dicts(lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures)

    with open('../predefined_mapping.json', 'r', encoding='utf-8') as f:
        predef_map = json.load(f)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    final_list = total_checker(lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, predef_map, entity, hadm_id, admission_str, charttime_str, conn, cursor, sets, lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, drug_list, d_icd_diag_lower_label_itemid_dict, d_icd_diag_keys, d_icd_proc_lower_label_itemid_dict, d_icd_proc_keys)

    return final_list


def search_tool(entity, pre_dict, lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, drug_list, sets, index, row_id, hadm_id, admission_str, charttime_str):
    try:
        final_values = process_data(entity, pre_dict, lab_id, d_items, prescriptions, d_icd_diagnoses, d_icd_procedures, drug_list, sets, index, row_id, hadm_id, admission_str, charttime_str)
        return final_values
            
    except KeyboardInterrupt:
        print()

def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  try:
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature= 0.1
    )
    return completion.choices[0].message.content
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  try:
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature= 0.1
    )
    return completion.choices[0].message.content
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def gemini_inference_for_search(prompt, model):
    gemini_response = model.models.generate_content(model="gemini-2.5-flash", 
                                                    contents=prompt,
                                                    config=types.GenerateContentConfig(
                                                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
    
    output = gemini_response.text
    return output

def find_items_from_dicts(subnote, table_name, target_entity, dfs, vertex_model):
    prescriptions_subset = dfs["prescriptions"]
    outputevents_subset = dfs["outputevents"]
    procedureevents_mv_subset = dfs["procedureevents_mv"]
    microbiologyevents_subset = dfs["microbiologyevents"]
    inputevents_cv_subset = dfs["inputevents_cv"]
    inputevents_mv_subset = dfs["inputevents_mv"]
    chartevents_subset = dfs["chartevents"]
    procedures_icd_subset = dfs['procedures_icd']
    diagnoses_icd_subset = dfs['diagnoses_icd']

    d_items_subset = dfs["d_items"]
    d_labitems_subset = dfs["d_labitems"]
    d_icd_diagnoses = dfs['d_icd_diagnoses']
    d_icd_procedures = dfs['d_icd_procedures']

    if 'lab' in table_name:
        items = list(d_labitems_subset['LABEL'])
    elif 'pres' in table_name:
        items = list(prescriptions_subset['DRUG'])
    elif 'icd' in table_name and 'diag' in table_name:
        items = list(d_icd_diagnoses[d_icd_diagnoses['ICD9_CODE'].isin(list(diagnoses_icd_subset['ICD9_CODE']))]['SHORT_TITLE'])
    elif 'icd' in table_name and 'proc' in table_name:
        items = list(d_icd_procedures[d_icd_procedures['ICD9_CODE'].isin(list(procedures_icd_subset['ICD9_CODE']))]['SHORT_TITLE'])

    else:
        d_items_list = list(outputevents_subset['ITEMID']) + list(procedureevents_mv_subset['ITEMID']) + list(microbiologyevents_subset['ORG_ITEMID']) + list(microbiologyevents_subset['SPEC_ITEMID']) + list(inputevents_cv_subset['ITEMID']) + list(inputevents_mv_subset['ITEMID']) + list(chartevents_subset['ITEMID'])
        d_items_list = [a for a in d_items_list if str(a) != 'nan']
        d_items_list = list(set(d_items_list))
        items = list(d_items_subset[d_items_subset['ITEMID'].isin(d_items_list)]['LABEL'])

    items = list(set(items))

    if len(items) > 50:
        chunk_lists = [items[i:i + 50] for i in range(0, len(items), 50)]
    else:
        chunk_lists = [items]
    
    new_items = []
    for i in range(2):
        for chunk_list in chunk_lists:
            items_find_prompt = f"Based on the following subnote, which explicitly contains the entity '{target_entity}', find all items that are related in meaning to '{target_entity}' from the list below.\nUse the content of the subnote as the primary context when identifying related items.\nSubnote:\n{subnote}\n\nItem list: {chunk_list}"
            item_results = gemini_inference_for_search(items_find_prompt, vertex_model)
            for item_ in chunk_list:
                if str(item_) in item_results:
                    new_items.append(item_)
    new_items = list(set(new_items))

    if 'lab' in table_name:
        return d_labitems_subset[d_labitems_subset['LABEL'].isin(new_items)]
    elif 'pres' in table_name:
        return prescriptions_subset[prescriptions_subset['DRUG'].isin(new_items)]
    elif 'icd' in table_name and 'diag' in table_name:
        return d_icd_diagnoses[d_icd_diagnoses['SHORT_TITLE'].isin(new_items)]
    elif 'icd' in table_name and 'proc' in table_name:
        return d_icd_procedures[d_icd_procedures['SHORT_TITLE'].isin(new_items)]
    else:
        return d_items_subset[d_items_subset['LABEL'].isin(new_items)]


def find_all_items_from_dicts(dfs, tab_):
    labevents_subset = dfs['labevents']
    prescriptions_subset = dfs["prescriptions"]
    outputevents_subset = dfs["outputevents"]
    procedureevents_mv_subset = dfs["procedureevents_mv"]
    microbiologyevents_subset = dfs["microbiologyevents"]
    inputevents_cv_subset = dfs["inputevents_cv"]
    inputevents_mv_subset = dfs["inputevents_mv"]
    chartevents_subset = dfs["chartevents"]
    d_items_subset = dfs["d_items"]
    d_labitems_subset = dfs["d_labitems"]

    lab_items = list(labevents_subset['ITEMID'])
    pres_items = list(prescriptions_subset['DRUG'])
    
    if 'lab' in tab_:
        return d_labitems_subset[d_labitems_subset['ITEMID'].isin(list(lab_items))]
    elif 'pres' in tab_:
        return prescriptions_subset
    else:
        d_items_list = list(outputevents_subset['ITEMID']) + list(procedureevents_mv_subset['ITEMID']) + list(microbiologyevents_subset['ORG_ITEMID']) + list(microbiologyevents_subset['SPEC_ITEMID']) + list(inputevents_cv_subset['ITEMID']) + list(inputevents_mv_subset['ITEMID']) + list(chartevents_subset['ITEMID'])
        return d_items_subset[d_items_subset['ITEMID'].isin(d_items_list)]

def top_values(entity, table, top_value_dict):
    return top_value_dict[table][entity]
