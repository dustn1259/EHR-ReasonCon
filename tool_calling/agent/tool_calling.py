from dotenv import load_dotenv
from google import genai
from google.genai import types
from tool_env import TerminalSearchTool
import os
import ast
import json
import pickle
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import time

# EHR-ReasonCon dataset root (the PhysioNet bundle). Defaults to <release>/EHR_ReasonCon;
# override with the EHR_REASONCON_DIR env var (see check_dataset.py / README).
# Extra attempts for the final verification call when its answer is unparsable (see main loop)
VERIFICATION_RETRIES = int(os.getenv("VERIFICATION_RETRIES", "2"))

EHR_REASONCON_DIR = os.getenv(
    "EHR_REASONCON_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EHR_ReasonCon")),
).rstrip("/")


def init_vertex_model():
    load_dotenv()
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
    return client


def gemini_inference(model, model_name, system_prompt, prompt, must_have_items = [], retry_num=5):

    for i in range(retry_num):
        try:
            if "gemini-3-flash-preview" == model_name:
                gemini_response = model.models.generate_content(model="gemini-3-flash-preview", 
                                                contents=prompt, 
                                                config=types.GenerateContentConfig(
                                                        system_instruction=system_prompt,
                                                        thinking_config=types.ThinkingConfig(thinking_level="low")
                                                        ),)
            elif 'gemini-2.5-flash' == model_name:
                gemini_response = model.models.generate_content(model="gemini-2.5-flash", 
                                                                contents=prompt, 
                                                                config=types.GenerateContentConfig(
                                                                    system_instruction=system_prompt))
            elif 'gemini-2.5-flash-noreason' == model_name:
                gemini_response = model.models.generate_content(model="gemini-2.5-flash", 
                                                                contents=prompt, 
                                                                config=types.GenerateContentConfig(
                                                                    system_instruction=system_prompt,
                                                                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            elif 'gemini-2.5' in model_name:
                gemini_response = model.models.generate_content(model="gemini-2.5-pro", 
                                                                contents=prompt,
                                                                config=types.GenerateContentConfig(
                                                                    system_instruction=system_prompt))
            elif 'gemini-2.0' in model_name:
                gemini_response = model.models.generate_content(model="gemini-2.0-flash", 
                                                                contents=prompt,
                                                                config=types.GenerateContentConfig(
                                                                    system_instruction=system_prompt))
            
            if not gemini_response.text:
                retry = 1
                continue

            retry = 0
            for must_item in must_have_items:
                if must_item not in gemini_response.text:
                    retry = 1
                    break
                must_content = gemini_response.text.split(must_item)[1].split('\n')[0].strip()
                if len(must_content) == 0:
                    retry = 1
                    break
                
                if must_item == 'The table to check':
                    retry = 1
                    # case-insensitive: the model may echo "Labevents"/"Prescriptions" as spelled in the note
                    tab_output = gemini_response.text.split('The table to check')[1].lower()
                    for tab in ["chartevents", "labevents", "prescriptions", "inputevents_mv", "inputevents_cv", "microbiologyevents", "procedureevents_mv", "outputevents", "diagnoses_icd", "procedures_icd"]:
                        if tab in tab_output:
                            retry = 0
                            break
            
            if retry == 0:
                break
        except:
            time.sleep(10)
            pass

    output = gemini_response.text.split('Query)')[0].split('(Tool Calling Result)')[0].strip()
    prompt = prompt.split('Output format)')[0].strip()
    prompt += "\n(Your Answer)\n" + output

    if "\n\nImportant instructions:\n-" in prompt and "\n\n---" in prompt:
        prompt = prompt.split("\n\nImportant instructions:\n-")[0] + prompt.split("\n\nImportant instructions:\n-")[1].split('\n\n---')[1]

    return output, prompt


def load_system_prompt():
    with open('./prompt/tool_calling_system.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    return system_prompt


def load_init_system_prompt():
    with open('./prompt/init_tool_calling_system.txt', 'r', encoding='utf-8') as f:
        init_system_prompt = f.read()
    return init_system_prompt


def load_verification_system_prompt():
    with open('./prompt/verification_system.txt', 'r', encoding='utf-8') as f:
        verification_system = f.read()
    return verification_system


def init_tool_select_prompt(linenum, entity, value, note, time):
    with open('./prompt/init_tool_select.txt', 'r', encoding='utf-8') as f:
        init_tool_select_prompt = f.read()
    if value == "":
        value = 'None'
    init_tool_select_prompt = init_tool_select_prompt.replace('<<line_number>>', linenum).replace('<<candidate_entity>>', entity).replace('<<candidate_value>>', value).replace('<<clinical_note>>', note).replace('<<TIME>>', time)
    return init_tool_select_prompt


def already_checked(recent_results, linenum, entity, value, note, model, model_name):
    with open('./prompt/already_checked.txt', 'r', encoding='utf-8') as f:
        already_checked_prompt = f.read()
    if value == "":
        value = 'None'
    if recent_results == "":
        recent_results = 'None'
    already_checked_prompt = already_checked_prompt.replace('<<line_number>>', linenum).replace('<<candidate_entity>>', entity).replace('<<candidate_value>>', value).replace('<<clinical_note>>', note).replace('<<already_events>>', recent_results)
    

    if "gemini-3-flash-preview" == model_name:
        gemini_response = model.models.generate_content(model="gemini-3-flash-preview", 
                                                contents=already_checked_prompt, 
                                                config=types.GenerateContentConfig(
                                                        system_instruction=system_prompt,
                                                        thinking_config=types.ThinkingConfig(thinking_level="low")
                                                        ),)
    elif 'gemini-2.5-flash' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.5-flash", 
                                                            contents=already_checked_prompt, 
                                                            config=types.GenerateContentConfig(
                                                                system_instruction=system_prompt))

    elif 'gemini-2.5' in model_name:
            gemini_response = model.models.generate_content(model="gemini-2.5-pro", 
                                                            contents=already_checked_prompt,
                                                            config=types.GenerateContentConfig(
                                                                system_instruction=system_prompt))
    
    elif 'gemini-2.0' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.0-flash", 
                                                        contents=already_checked_prompt,
                                                        config=types.GenerateContentConfig(
                                                            system_instruction=system_prompt))
    if 'yes' in gemini_response.text.split(':')[-1].lower():
        return True, already_checked_prompt + '\n' + gemini_response.text
    else:
        return False, already_checked_prompt + '\n' + gemini_response.text


def v1_data_processing(label_data):
    data_list = label_data[list(label_data)[0]]

    sorted_data = sorted(data_list,key=lambda x: int(list(x.values())[0]['position']))

    data_line_to_labeled = {}

    for dat in sorted_data:
        for skip_key in ['all_queries', 'tag', 'almost_consistent', 'consistency_check_path', 'medical_knowledge_source', 'commonsense_medical_none', 'entity_type', 'source', 'Check_DB']:
            if skip_key in list(dat[list(dat)[0]]):
                dat[list(dat)[0]].pop(skip_key)
        if 'data' not in dat[list(dat)[0]]:
            continue
        if 'label' in dat[list(dat)[0]]['data'][0]:
            if "incon" in dat[list(dat)[0]]['data'][0]['label'].lower():
                dat[list(dat)[0]]['data'][0]['label'] = "inconsistent"
            else:
                dat[list(dat)[0]]['data'][0]['label'] = "consistent"
                try:
                    dat[list(dat)[0]]['data'][0].pop('inconsistent_columns')
                except:
                    pass
                try:
                    dat[list(dat)[0]]['data'][0].pop('errors')
                except:
                    pass
        new_line = f"{list(dat)[0]} {dat[list(dat)[0]]}"
        new_line = new_line.replace("'position'", "'Line number'")

        try:
            data_line_to_labeled[int(dat[list(dat)[0]]['position'])].append(new_line.strip())
        except:
            data_line_to_labeled[int(dat[list(dat)[0]]['position'])] = [new_line.strip()]

    return data_line_to_labeled

def load_ner_output(hadm_id, id, note_type, testval, nermethod, dataver, model_name, model):
    try:
        with open(f'../../entity_extraction/output/{nermethod}ner_time_result_{note_type}_{testval}_{id}_v2_{model_name}_2.pkl', 'rb') as f:
            ner_output = pickle.load(f)
    except:
        with open(f'../../entity_extraction/output/{nermethod}ner_time_result_{note_type}_{testval}_{id}_v2_2.pkl', 'rb') as f:
            ner_output = pickle.load(f)
        
    ner_dict = {}
    for dat in ner_output:
        line_num = dat[0]
        ent_val = dat[-2]
        ent_val_splitted = ent_val.split(':')
        if len(ent_val_splitted) > 2:
            ent_val_splitted = [ent_val_splitted[0], ':'.join(ent_val_splitted[1:])]
        time = dat[-1][0]

        data_point_list = []
        for dat_p in ent_val_splitted:
            data_point_list.append(dat_p.replace('{', '').replace('}', '').strip())
        data_point_list.append(time)
        try:
            ner_dict[line_num].append(data_point_list)
        except:
            ner_dict[line_num] = [data_point_list]

    if dataver == 'v1':
        if "val" in testval:
            test_val_dir = "valid"
        else:
            test_val_dir = "test"
        with open(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/EHRReasonCon_{str(hadm_id)}_data.json', 'r', encoding='utf-8') as f:
            v1_data = json.load(f)
        processed_v1_data = v1_data_processing(v1_data)

        new_ner_dict = {}
        with open('./prompt/v1_filtering.txt', 'r', encoding='utf-8') as f:
            v1_filtering_prompt = f.read()
        for line_, ents_ in ner_dict.items():
            datas = []
            if int(line_)-1 in list(processed_v1_data):
                datas += processed_v1_data[int(line_)-1]
            if int(line_) in list(processed_v1_data):
                datas += processed_v1_data[int(line_)]
            if int(line_)+1 in list(processed_v1_data):
                datas += processed_v1_data[int(line_)+1]
            if len(datas) == 0:
                continue
            ner_result = ""
            for ents_i in range(len(ents_)):
                ner_result += f"{ents_i}. {ents_[ents_i][:2]}\n"
            v1_scope = ""
            for data_i in range(len(datas)):
                v1_scope += f"{datas[data_i]}\n"
            
            v1_filter_line = v1_filtering_prompt.replace('<<V1_ENTS>>', v1_scope.strip()).replace('<<LINE_ENTS>>', ner_result.strip())
            gemini_response = model.models.generate_content(model="gemini-2.5-flash", contents=v1_filter_line).text
            index_list = gemini_response.split('[')[-1].split(']')[0]

            new_ent_list = []
            if len(index_list) > 0 and len(ents_) == 1:
                new_ent_list.append(ents_[0])
            else:
                for index_ in index_list.split(','):
                    if index_.strip() == "":
                        continue
                    new_ent_list.append(ents_[int(index_.strip())])

            if new_ent_list == []:
                continue
            new_ner_dict[line_] = new_ent_list
        return new_ner_dict
    return ner_dict

def save_hadm_id(id, note_type, testval):
    if testval == 'val':
        test_val_dir = 'valid'
    else:
        test_val_dir = 'test'
    try:
        df = pd.read_csv(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/{note_type}_{testval}.csv')
        hadm_id = df['HADM_ID'][int(id)]
        with open("../current_hadm_id.txt", "w", encoding="utf-8") as f:
            f.write(f"{note_type}, {testval}, {hadm_id}")
        return True, hadm_id
    except:
        return False, ""

def load_note_subsection(id, note_type, testval, model_name):
    try:
        with open(f'../../entity_extraction/output/subtext_divided_{note_type}_{testval}_{id}_{model_name}_2.pkl', 'rb') as f:
            note_subsection = pickle.load(f)
    except:
        with open(f'../../entity_extraction/output/subtext_divided_{note_type}_{testval}_{id}_2.pkl', 'rb') as f:
            note_subsection = pickle.load(f)
    return note_subsection

def load_d_p_result(id, note_type, testval, nermethod, dataver, model_name):
    try:
        with open(f'../../entity_extraction/output/{nermethod}d_p_concheck_result_{note_type}_{testval}_{id}_v2_{model_name}.pkl', 'rb') as f:
            d_p_result = pickle.load(f)
        return d_p_result
    except:
        return {}


def make_prompt_entries(prompt, tool, categories):
    cond_dict = {"Item_Search": {"The entity to search": ""}, 
            "Top_Values_for_Entities": {"The table to check": " (Available tables (Choose only one table): chartevents, labevents, outputevents, prescriptions)", 
                                        "The item to check": ""}, 
            "Semantic_Search": {"The table to check": " (Available tables: ditems(chartevents, inputevents, outputevents, procedureevents_mv, microbiologyevents), labitems, prescriptions, d_icd_diagnoses, d_icd_procedures)", 
                                "The entity to search": ""}, 
            "Table_Value_Time": {"The table to check": " (Available tables (Choose only one table): chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, outputevents)", 
                                "Value Range": "", 
                                "Time Range": ""}, 
            "Table_Category_Time": {"The table to check": f" (Available tables: {categories[0]})", 
                                    "Category": categories[1], 
                                    "Time Range": ""},
            "Table_Time": {"The table to check": " (Available tables (Choose only one table): chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, microbiologyevents, procedureevents_mv, outputevents, diagnoses_icd, procedures_icd)", 
                            "Time Range": ""},
            "Table_Selected_Item_Time": {"The table to check": " (Available tables (Choose only one table): chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, microbiologyevents, procedureevents_mv, outputevents, diagnoses_icd, procedures_icd)", 
                                            "The item to check": " (Please select only one item, as only one can be checked at a time.)", 
                                            "Time Range": ""}, 
            "Table_Selected_Item_Value_Time": {"The table to check": " (Available tables (Choose only one table): chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, outputevents)", 
                                                "The item to check": " (Please select only one item, as only one can be checked at a time.)", 
                                                "Value Range": "", 
                                                "Time Range": ""}}
    
    fill_cond_prompt = f"\n\nQuery) Please provide answers to the following entries for {tool}."
    output_prompt = "\n\nOutput format)"
    contents = cond_dict[tool]
    for k, v in contents.items():
        fill_cond_prompt += f"\n{k}{v}:"
        output_prompt += f"\n{k}:"
    
    prompt += fill_cond_prompt + output_prompt
    return prompt, cond_dict


def make_prompt_final(tool_result, prompt):
    with open('./prompt/options.txt', 'r', encoding='utf-8') as f:
        verification = f.read()
    try:
        prompt = f'{prompt}\n\n(Tool Calling Result)\n{tool_result.to_string().strip()}\n\n{verification}'
    except:
        if len(str(tool_result)) == 0:
            tool_result = ""
        prompt = f'{prompt}\n\n(Tool Calling Result)\n{str(tool_result).strip()}\n\n{verification}' 

    return prompt


def make_prompt_filter_res(tool_result, prompt):
    cols = tool_result.columns
    data_prompt = ""
    if 'DRUG' in cols:
        drgs = list(tool_result['DRUG'].unique())
        data_prompt += f"\n\nDRUG: {drgs}"
    if 'LABEL' in cols:
        labels = list(tool_result['LABEL'].unique())
        data_prompt += f"\n\nLABEL: {labels}"
    if 'SPEC_TYPE_DESC' in cols:
        spectypes = list(tool_result['SPEC_TYPE_DESC'].unique())
        data_prompt += f"\n\nSPEC_TYPE_DESC: {spectypes}"
    if 'ORG_NAME' in cols:
        orgnames = list(tool_result['ORG_NAME'].unique())
        data_prompt += f"\n\nORG_NAME: {orgnames}"
    if 'VALUE' in cols:
        values = list(tool_result['VALUE'].unique())
        data_prompt += f"\n\nVALUE: {values}"

    with open('./prompt/filter_prompt.txt', 'r', encoding='utf-8') as f:
        filter_prompt = f.read()
    filter_prompt = filter_prompt.replace('<<COLUMNS>>', data_prompt)
    prompt = f'{prompt}\n\n{filter_prompt}' 

    return prompt


def make_prompt_verification(prompt):
    prompt = f'{prompt}\n\nAssess whether the information explicitly stated in the clinical note is fully consistent with the structured data. (Make your judgment based on clinical meaning.)' 
    return prompt


def tool_process(tool):
    if 'Item_Search' in tool:
        tool = 'Item_Search'
    elif 'Semantic_Search' in tool:
        tool = 'Semantic_Search'
    elif 'Top_Values_for_Entities' in tool:
        tool = 'Top_Values_for_Entities'
    elif 'Table_Value_Time' in tool:
        tool = 'Table_Value_Time'
    elif 'Table_Category_Time' in tool:
        tool = 'Table_Category_Time'
    elif 'Table_Time' in tool:
        tool = 'Table_Time'
    elif 'Table_Selected_Item_Time' in tool:
        tool = 'Table_Selected_Item_Time'
    elif 'Table_Selected_Item_Value_Time' in tool:
        tool = 'Table_Selected_Item_Value_Time'
    else:
        tool = 'error'
    return tool


def check_validity(tool, table_name):
    if "Item_Search".lower() in tool.lower() or 'Table_Category_Time'.lower() in tool.lower():
        return True
    
    tool_to_tabs = {
        "Top_Values_for_Entities": "chartevents, labevents, outputevents, prescriptions",
        "Semantic_Search": "ditems, labitems, prescriptions, d_icd_diagnoses, d_icd_procedures",
        "Table_Value_Time": "chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, outputevents", 
        "Table_Time": "chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, microbiologyevents, procedureevents_mv, outputevents, diagnoses_icd, procedures_icd", 
        "Table_Selected_Item_Time": "chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, microbiologyevents, procedureevents_mv, outputevents, diagnoses_icd, procedures_icd", 
        "Table_Selected_Item_Value_Time": "chartevents, labevents, prescriptions, inputevents_mv, inputevents_cv, outputevents", 
    }

    if table_name.lower() not in tool_to_tabs[tool]:
        return False
    return True
    

def find_related_info(model, model_name, line_num, ent, val, new_note, ent_tool_result):
    item_search_result = ""
    semantic_search_result = ""
    lab_or_pres = "ditems"
    for tool_history in ent_tool_result:
        if 'lab' in str(tool_history["query"]).lower():
            lab_or_pres = 'labitems'
        if 'presc' in str(tool_history["query"]).lower():
            lab_or_pres = 'prescriptions'
        if 'diag' in str(tool_history["query"]).lower():
            lab_or_pres = 'd_icd_diagnoses'
        if '_procedure' in str(tool_history["query"]).lower():
            lab_or_pres = 'd_icd_procedures'
        if "search" in tool_history["query"]["tool"].lower():
            if "item" in tool_history["query"]["tool"].lower():
                item_search_result += str(tool_history["query"])
                item_search_result += "\n" + tool_history["result"]

            elif "semantic" in tool_history["query"]["tool"].lower():
                semantic_search_result += str(tool_history["query"])
                semantic_search_result += "\n" + tool_history["result"]

    if item_search_result == "":
        new_search_param = {'tool': "Item_Search", 'subnote': new_note, 'params': {"The entity to search": ent}}
        tool_result = tool_env.tool_call(new_search_param)
        try:
            tool_result = tool_result.to_string().strip()
        except:
            tool_result = str(tool_result).strip()
        item_search_result += str(new_search_param)
        item_search_result += "\n" + tool_result
    
    if semantic_search_result == "":
        new_search_param = {'tool': "Semantic_Search", 'subnote': new_note, 'params': {"The table to check": lab_or_pres, "The entity to search": ent}}
        tool_result = tool_env.tool_call(new_search_param)
        try:
            tool_result = tool_result.to_string().strip()
        except:
            tool_result = str(tool_result).strip()
        semantic_search_result += str(new_search_param)
        semantic_search_result += "\n" + tool_result

    search_result = item_search_result + '\n\n' + semantic_search_result

    with open('./prompt/incon_check_related_info.txt', 'r', encoding='utf-8') as f:
        incon_check_related_info_prompt = f.read()

    incon_check_related_info_prompt = incon_check_related_info_prompt.replace('<<line_number>>', line_num).replace('<<candidate_entity>>', ent).replace('<<candidate_value>>', val).replace('<<clinical_note>>', new_note).replace('<<search_result>>', search_result)


    if "gemini-3-flash-preview" == model_name:
        gemini_response = model.models.generate_content(model="gemini-3-flash-preview", 
                                                contents=incon_check_related_info_prompt, 
                                                config=types.GenerateContentConfig(
                                                        system_instruction=system_prompt,
                                                        thinking_config=types.ThinkingConfig(thinking_level="low")
                                                        ),)
    elif 'gemini-2.5-flash' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.5-flash", 
                                                            contents=incon_check_related_info_prompt, 
                                                            config=types.GenerateContentConfig(
                                                                system_instruction=system_prompt))
    elif 'gemini-2.5' in model_name:
            gemini_response = model.models.generate_content(model="gemini-2.5-pro", 
                                                            contents=incon_check_related_info_prompt,
                                                            config=types.GenerateContentConfig(
                                                                system_instruction=system_prompt))
    elif 'gemini-2.0' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.0-flash", 
                                                        contents=incon_check_related_info_prompt,
                                                        config=types.GenerateContentConfig(
                                                            system_instruction=system_prompt))

    if 'Items:' in gemini_response.text:
        items_text = gemini_response.text.split('Items:')[1].strip()
    elif 'items:' in gemini_response.text:
        items_text = gemini_response.text.split('items:')[1].strip() 
    if ';;' in items_text:
        item_list = items_text.split(';;')
    elif ';' in items_text:
        item_list = items_text.split(';')
    elif ',' in items_text:
        item_list = items_text.split(',')
        item_list.append(items_text)
    else:
        item_list = [items_text]
    return item_list


def top_values_for_items(tool_env, items):
    prescription_trigg = False
    item_to_topvalues = {}
    for item_ in items:
        item_to_topvalues[item_] = []
        for tab in ["chartevents", "labevents", "outputevents", "prescriptions"]:
            new_search_param = {}
            new_search_param['tool'] = "Top_Values_for_Entities"
            new_search_param['params'] = {"The table to check": tab}
            new_search_param['params']['The item to check'] = item_.strip()
            try:
                tool_result = tool_env.tool_call(new_search_param)
                item_to_topvalues[item_.strip()] += tool_result
                item_to_topvalues[item_.strip()] = list(set(item_to_topvalues[item_.strip()]))
                if tab == "prescriptions":
                    prescription_trigg = True
            except:
                pass

    return item_to_topvalues, prescription_trigg


def cannot_verify(model, model_name, line_num, ent, val, new_note, item_to_topvalues):
    system_prompt = "You are an expert in determining whether the event described in the clinical note is related to the given information (Item Related Information). Read carefully and provide your evaluation according to the output format."
    with open('./prompt/check_verifiable.txt', 'r', encoding='utf-8') as f:
        check_verifiable_prompt = f.read()

    item_to_top_prompt = ""
    for k, v in item_to_topvalues.items():
        item_to_top_prompt += f"Item: {k}, Values: {v}\n\n"

    check_verifiable_prompt = check_verifiable_prompt.replace('<<line_number>>', line_num).replace('<<candidate_entity>>', ent).replace('<<candidate_value>>', val).replace('<<clinical_note>>', new_note).replace('<<topvalues>>', item_to_top_prompt.strip())

    if "gemini-3-flash-preview" == model_name:
        gemini_response = model.models.generate_content(model="gemini-3-flash-preview", 
                                                contents=check_verifiable_prompt, 
                                                config=types.GenerateContentConfig(
                                                        system_instruction=system_prompt,
                                                        thinking_config=types.ThinkingConfig(thinking_level="low")
                                                        ),)
    elif 'gemini-2.5-flash' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.5-flash", 
                                                        contents=check_verifiable_prompt, 
                                                        config=types.GenerateContentConfig(
                                                            system_instruction=system_prompt))
    elif 'gemini-2.5' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.5-pro", 
                                                        contents=check_verifiable_prompt,
                                                        config=types.GenerateContentConfig(
                                                            system_instruction=system_prompt))

    
    elif 'gemini-2.0' in model_name:
        gemini_response = model.models.generate_content(model="gemini-2.0-flash", 
                                                        contents=check_verifiable_prompt,
                                                        config=types.GenerateContentConfig(
                                                            system_instruction=system_prompt))

    if 'Result:' in gemini_response.text:
        check_text = gemini_response.text.split('Result:')[1].strip()
    elif 'result:' in gemini_response.text:
        check_text = gemini_response.text.split('result:')[1].strip() 
    else:
        check_text = gemini_response.text

    check_verifiable_prompt += "\n" + gemini_response.text

    if 'yes' in check_text.lower():
        return False, check_verifiable_prompt
    return True, check_verifiable_prompt

def adm_dis_whole_range(adm_date, dis_date):
    adm_before = adm_date - timedelta(days=1)
    adm_after = adm_date + timedelta(days=1)
    dis_before = dis_date - timedelta(days=1)
    dis_after = dis_date + timedelta(days=1)

    adm_range = f"{adm_before.strftime('%Y-%m-%d')} to {adm_after.strftime('%Y-%m-%d')}"
    dis_range = f"{dis_before.strftime('%Y-%m-%d')} to {dis_after.strftime('%Y-%m-%d')}"
    whole_range = f"{adm_before.strftime('%Y-%m-%d')} to {dis_after.strftime('%Y-%m-%d')}"
    
    return adm_range, dis_range, whole_range

def chart_date_range(id, note_type, test_val):
    if test_val == 'val':
        test_val_dir = 'valid'
    else:
        test_val_dir = 'test'
    
    df = pd.read_csv(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/{note_type}_{test_val}.csv')
    chart_date = df['CHARTDATE'][int(id)]
    date = datetime.strptime(chart_date, '%Y-%m-%d')
    start_date = date - timedelta(days=1)
    end_date = date + timedelta(days=1)

    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"


if __name__ == '__main__':

    _idx = os.getenv("NOTE_IDX")  # e.g. NOTE_IDX=6 to run one note; unset = all
    id_range = [int(_idx), int(_idx)] if _idx is not None else [0,100] # {[start_id, last_id] e.g., [0,0], [0,5], for all: [0,100]}
    NOTE_TYPE = os.getenv("NOTE_TYPE", "discharge")  # {"discharge", "physician", "nursing"}
    TEST_VAL = os.getenv("TEST_VAL", "val") # {"test", "val"}
    NER_METHOD = os.getenv("NER_METHOD", "cluster") # {"hyperthr", "bert", "cluster", "trgem", "gold", "router", "checkehr"}
    DATA_VERSION = os.getenv("DATA_VERSION", "v2")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini")  # {"gemini"}

    for note_id in range(100):
        if note_id < id_range[0] or note_id > id_range[1]:
            continue
        ID = str(note_id)

        tf, hadm_id = save_hadm_id(ID, NOTE_TYPE, TEST_VAL)
        if not tf:
            continue

        vertex_model = init_vertex_model()
        system_prompt = load_system_prompt()
        init_system_prompt = load_init_system_prompt()
        verification_system_prompt = load_verification_system_prompt()
        tool_env = TerminalSearchTool()
        tool_env.tool_init(vertex_model=vertex_model)

        chart_time_range_ = chart_date_range(ID, NOTE_TYPE, TEST_VAL)
        subnotes = load_note_subsection(ID, NOTE_TYPE, TEST_VAL, MODEL_NAME)
        ner_output = load_ner_output(hadm_id, ID, NOTE_TYPE, TEST_VAL, NER_METHOD, DATA_VERSION, MODEL_NAME, vertex_model)
        try:
            with open(f'./result/concheck_result_{ID}_{NOTE_TYPE}_{TEST_VAL}_{NER_METHOD}_{DATA_VERSION}_2.json', 'r') as f:
                consistency_result = json.load(f)
        except:
            consistency_result = load_d_p_result(ID, NOTE_TYPE, TEST_VAL, NER_METHOD, DATA_VERSION, MODEL_NAME)
        available_cats = tool_env.category_by_table()
        note_check_memory = deque(maxlen=5)
        for _, note in subnotes.items():
            for line in note.split('\n'):
                line_num = line.split(':\t')[0].strip()
                try:
                    entity_data = ner_output[line_num]
                except:
                    continue

                try:
                    a = consistency_result[line_num]
                    continue
                except:
                    consistency_result[line_num] = []

                for ent, val, time_ in entity_data:
                    checked_items = []
                    ent_tool_result = []
                    new_note = ""
                    for l in note.split('\n'):
                        l_num = l.split('   ')[0].strip()
                        if line_num == l_num:
                            if val == "":
                                l = l.replace(ent, f"**{ent}**")
                            else:
                                l = l.replace(ent, f"**{ent}**").replace(val, f"**{val}**")
                        new_note += l + '\n'
                    
                    recent_results = "\n\n".join(note_check_memory)

                    already_, skip_prompt = already_checked(recent_results, line_num, ent, val, new_note, vertex_model, 'gemini-2.5-flash')
                    if already_:
                        consistency_result[line_num].append({'entity': ent, 'consistency': "skipped", 'tool_call_process': skip_prompt})
                        continue

                    tool_select_prompt = init_tool_select_prompt(line_num, ent, val, new_note, time_)
                    gemini_model_name = 'gemini-2.5-flash'

                    for tn in range(10):
                        gemini_response, tool_select_prompt_ = gemini_inference(vertex_model, gemini_model_name, init_system_prompt, tool_select_prompt)
                        try:
                            tool = gemini_response.split("Selected tool:")[1].split('\n')[0].strip()
                        except:
                            continue
                        tool = tool_process(tool)
                        if 'error' not in tool.lower():
                            tool_select_prompt = tool_select_prompt_
                            break

                    for _ in range(10):
                        tool_select_prompt_, cond_dict = make_prompt_entries(tool_select_prompt, tool, available_cats)
                        gemini_response, tool_select_prompt_ = gemini_inference(vertex_model, gemini_model_name, system_prompt, tool_select_prompt_, list(cond_dict[tool]))
                        if tool == 'Item_Search' and "The entity to search" not in gemini_response:
                            gemini_response = f"The entity to search:{gemini_response.strip()}"
                        val_spl = gemini_response.split('\n')
                        
                        search_param = {'tool': tool, 'subnote': note, 'params': {}}
                        
                        for val_fill in val_spl:
                            for col in cond_dict[tool]:
                                if col in val_fill:
                                    if col+':' in val_fill:
                                        proc_val = val_fill.split(col+':')[1].strip()
                                    
                                    if '<=' in proc_val:
                                        proc_val = proc_val.replace('<=', '<')

                                    if col == 'The table to check':
                                        # get_tables() matches table names exactly in lowercase; the model may echo
                                        # the capitalised spellings used in the note/catalog (e.g. "Labevents")
                                        proc_val = proc_val.strip().lower()

                                    if tool.lower() == 'Semantic_Search'.lower():
                                        if col == 'The table to check':
                                            if proc_val.lower() in 'chartevents, inputevents, outputevents, procedureevents_mv, microbiologyevents':
                                                proc_val = "ditems"
                                            if 'ditems' in proc_val or 'd_items' in proc_val:
                                                proc_val = 'ditems'
                                            if 'diag' in proc_val:
                                                proc_val = 'd_icd_diagnoses'
                                            if 'icd' in proc_val and 'proc' in proc_val:
                                                proc_val = 'd_icd_procedures'
                                    if 'time range' in col.lower():
                                        temp_adm_range, temp_dis_range, temp_whole_range = adm_dis_whole_range(tool_env.admitdate, tool_env.dischdate)
                                        
                                        if NOTE_TYPE == 'discharge': 
                                            if 'today' in proc_val.lower() or len(proc_val.strip()) == 0 or 'all' in proc_val.lower() or 'whole' in proc_val.lower() or 'entire' in proc_val.lower() or 'none' in proc_val.lower():
                                                proc_val = temp_whole_range
                                            elif 'adm' in proc_val.lower():
                                                proc_val = temp_adm_range
                                            elif 'dis' in proc_val.lower():
                                                proc_val = temp_dis_range
                                        else:
                                            if 'today' in proc_val.lower() or len(proc_val.strip()) == 0 or 'all' in proc_val.lower() or 'whole' in proc_val.lower() or 'entire' in proc_val.lower() or 'none' in proc_val.lower() or 'chart' in proc_val.lower():
                                                proc_val = chart_time_range_
                                            elif 'adm' in proc_val.lower():
                                                proc_val = temp_adm_range
                                    search_param['params'][col] = proc_val


                        if "The table to check" in search_param['params']:
                            if not check_validity(tool, search_param['params']["The table to check"]):
                                continue
                        if search_param['params'] == {}:
                            continue

                        # suspect point
                        tool_select_prompt = tool_select_prompt_

                        ### Tool call
                        try:
                            tool_result = tool_env.tool_call(search_param)
                        except:
                            if 'The item to check' in search_param['params']:
                                if " (" in search_param['params']['The item to check']:
                                    search_param['params']['The item to check'] = search_param['params']['The item to check'].replace(' (', '  (')
                            try:
                                tool_result = tool_env.tool_call(search_param)
                            except:
                                tool_result = "INVALID INPUT"
                        
                        if "The item to check" in search_param['params']:
                            checked_items.append(search_param['params']["The item to check"])

                        ### record tool calling history
                        try:
                            ent_tool_result.append({"query": search_param, "result": tool_result.to_string().strip()})
                        except:
                            if len(str(tool_result)) == 0:
                                tool_result = "None"
                            ent_tool_result.append({"query": search_param, "result": str(tool_result).strip()})

                        if len(str(tool_result)) > 10000 and tool not in ['Semantic_Search', 'Item_Search', 'Table_Selected_Item_Time', 'Table_Selected_Item_Value_Time']:
                            try:
                                tool_select_prompt = make_prompt_filter_res(tool_result, tool_select_prompt)
                                gemini_response, tool_select_prompt = gemini_inference(vertex_model, gemini_model_name, "You are an assistant that performs consistency checks between two data. Understand the context of the consistency check process and provide an answer to the final query.", tool_select_prompt)
                                column_name = gemini_response.split('Column:')[1].split('\n')[0].strip()
                                target_name = gemini_response.split('Target:')[1].split('\n')[0].strip()
                                if ';;' in column_name:
                                    column_name = column_name.split(';;')[0].strip()
                                target_list = target_name.split(';;')
                                filtered_tool_result = tool_result[tool_result[column_name].isin(target_list)]
                                tool_select_prompt = make_prompt_final(filtered_tool_result, tool_select_prompt)
                            except:
                                pass
                        else:
                            tool_select_prompt = make_prompt_final(tool_result, tool_select_prompt)
                        
                        gemini_response, tool_select_prompt_ = gemini_inference(vertex_model, gemini_model_name, system_prompt, tool_select_prompt)
                        if "Selected tool: " in gemini_response and 'option: 2' not in gemini_response.lower():
                            gemini_response += "\noption: 2"
                            tool_select_prompt_ += "\noption: 2"
                        
                        if "option: 1" in gemini_response.lower():
                            # Verification: ask once, retry the verification call alone up to
                            # VERIFICATION_RETRIES more times if the answer is missing or lacks the
                            # 'Consistency status' line. Never re-runs the tool-call step.
                            verification_prompt = make_prompt_verification(tool_select_prompt_)
                            con_res = ""
                            for v_try in range(1 + VERIFICATION_RETRIES):
                                try:
                                    gemini_response, verified_prompt = gemini_inference(vertex_model, "gemini-2.5-flash-noreason", verification_system_prompt, verification_prompt)
                                    con_res = gemini_response.split('Consistency status (Consistent/Inconsistent):')[1].split('\n')[0].strip()
                                except:
                                    con_res = ""
                                if con_res:
                                    tool_select_prompt = verified_prompt
                                    print('\n'+tool_select_prompt)
                                    break
                                print(f'Verification output unparsable (attempt {v_try+1}/{1+VERIFICATION_RETRIES})')
                            if not con_res:
                                # give up on this entity explicitly: a visible label instead of a silent gap
                                print('Verification failed; recording as "verification failed"')
                                consistency_result[line_num].append({'entity': ent, 'consistency': "verification failed", 'tool_call_process': verification_prompt})
                                break
                            if 'inconsistent' in con_res.lower():
                                print('Inconsistent')
                            else:
                                print('Consistent')
                            

                            # If inconsistent -> check whether it is inconsistent or it does not exist in the whole DB.
                            try:
                                is_val_float = float(val)
                            except:
                                if 'inconsistent' in con_res.lower():
                                    # item search, semantic search result -> whether there is an aligning entity
                                    related_items = find_related_info(vertex_model, 'gemini-2.5-flash', line_num, ent, val, new_note, ent_tool_result)
                                    related_items += checked_items
                                    related_items = list(set(related_items))
                                    item_to_topvalues, prescription_trigg = top_values_for_items(tool_env, related_items)
                                    if not prescription_trigg:
                                        non_verifiable, veri_prompt = cannot_verify(vertex_model, 'gemini-2.5-flash', line_num, ent, val, new_note, item_to_topvalues)
                                        if non_verifiable:
                                            consistency_result[line_num].append({'entity': ent, 'consistency': "outside the scope", 'tool_call_process': veri_prompt})
                                            break

                            consistency_result[line_num].append({'entity': ent, 'consistency': con_res, 'tool_call_process': tool_select_prompt})
                            
                            ent_info = tool_select_prompt.split('\n')[0]
                            ent_concheck_res = tool_select_prompt.split('Your Answer)')[-1].split('Evidence index')[0].strip()
                            note_check_memory.append(ent_info+'\n'+ent_concheck_res)
                            
                            break
                        
                        elif "option: 2" in gemini_response.lower():
                            try:
                                tool = gemini_response.split("Selected tool:")[1].split('\n')[0].strip()
                                tool = tool_process(tool)
                            except:
                                pass
                            if 'error' in tool.lower():
                                for tn in range(5):
                                    gemini_response, tool_select_prompt_ = gemini_inference(vertex_model, gemini_model_name, init_system_prompt, tool_select_prompt)
                                    try:
                                        tool = gemini_response.split("Selected tool:")[1].split('\n')[0].strip()
                                    except:
                                        pass
                                    tool = tool_process(tool)
                                    if 'error' not in tool.lower():
                                        break
                            
                            tool_select_prompt = tool_select_prompt_
                            
                            if len(str(tool_result)) > 10000:
                                tool_select_prompt = tool_select_prompt.replace(str(tool_result), "[Tool call result omitted]")

                        else:
                            pass

                    with open(f'./result/concheck_result_{ID}_{NOTE_TYPE}_{TEST_VAL}_{NER_METHOD}_{DATA_VERSION}_2.json', 'w') as f:
                        json.dump(consistency_result, f)
