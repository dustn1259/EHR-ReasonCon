import json
import re
import pandas as pd
import pickle
import os
from google.genai import types
from models import init_vertex_model, get_azure_response
from collections import deque
import time



# EHR-ReasonCon dataset root (the PhysioNet bundle). Defaults to <release>/EHR_ReasonCon;
# override with the EHR_REASONCON_DIR env var (see check_dataset.py / README).
EHR_REASONCON_DIR = os.getenv(
    "EHR_REASONCON_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EHR_ReasonCon")),
).rstrip("/")


def organized_preds(pred_result_name, model, model_name):
    if 'checkehr' in pred_result_name and 'checkehr_v2.json' not in pred_result_name:
        with open(pred_result_name, 'r') as f:
            checkehr = json.load(f)
        pred_json = {}
        for c_num, vals in checkehr.items():
            for val_ in vals:
                entity_name = list(val_)[0]
                result_dict = val_[entity_name]
                l_num = result_dict['position']
                consistency_ = result_dict['label']
                related_info = result_dict['table_cell']
                sql_ = result_dict['sql']
                reasoning_process = f'When I found the times or values related to entity "{entity_name}" written in the notes'
                for col_, cell_val in related_info.items():
                    if cell_val.lower() == "nan":
                        continue
                    reasoning_process += f", the value of {col_} was {cell_val}"
                if 'inconsist' in consistency_.lower():
                    reasoning_process += f". Using this information, I executed the following SQL query [{sql_}] on the database, and since the corresponding record did not exist, this is inconsistent."
                else:
                    reasoning_process += f". Using this information, I executed the following SQL query [{sql_}] on the database, and since the corresponding record existed, this is consistent."
                
                if str(l_num) in list(pred_json):
                    pred_json[str(l_num)].append({'entity': entity_name, 'consistency': consistency_, 'tool_call_process': reasoning_process})
                else:
                    pred_json[str(l_num)] = [{'entity': entity_name, 'consistency': consistency_, 'tool_call_process': reasoning_process}]
    else:
        with open(pred_result_name, 'r') as f:
            pred_json = json.load(f)
    
    with open("./prompt/eval_divide_system.txt", "r", encoding="utf-8") as f:
        divide_system = f.read()

    sorted_dict = dict(sorted(pred_json.items(), key=lambda x: int(x[0])))
    line_to_pred = {}
    skipped_line_to_pred = {}
    for id_, val in sorted_dict.items():
        for data_point in val:
            if data_point['consistency'] in ('outside the scope', 'verification failed', 'tool selection failed'):
                # 'verification failed' = Stage 2 gave up after VERIFICATION_RETRIES; there is no
                # verdict to judge, so it counts as no prediction (a recall miss), never as precision
                continue
            if "SHORT_TITLE or LONG_TITLE" in data_point['tool_call_process'].split('\n')[0]:
                temp_part = data_point['tool_call_process'].split('Given entity: ')[1].split('\n')[0]
                new_line = f"Line number: {id_} Candidate entity: {temp_part}"
                print("Line number:", id_, "Candidate entity: ", data_point['tool_call_process'].split('Given entity: ')[1].split('\n')[0])
            elif data_point['consistency'] =='skipped':
                new_line = data_point['tool_call_process'].split('if not, output "no".')[1].strip().split('\n')[0]
                new_line += data_point['tool_call_process'].split('Result: (yes or no)')[1]
                new_line = new_line.replace('\n\nResult: yes', "").replace('\nResult: yes', "")
                try:
                    skipped_line_to_pred[int(id_)].append(new_line)
                except:
                    skipped_line_to_pred[int(id_)] = [new_line]
                continue
            else:
                if 'Assess whether the information explicitly ' in data_point['tool_call_process'].split('\n')[0]:
                    data_point['tool_call_process'] = data_point['tool_call_process'].split('(Make your judgment based on clinical meaning.)')[1].strip()
                new_line = data_point['tool_call_process'].split('\n')[0]
                print(new_line)
            
            if "Given the following table, determine whether the specified entity is mentioned within the SHORT_TITLE or LONG_TITLE columns." in data_point['tool_call_process'].split('(Your Answer)')[-1].strip():
                p_d_new_line = data_point['tool_call_process'].split('(Your Answer)')[-1].strip()
                p_d_new_line = p_d_new_line.split('Reason:')[-1].strip()
                p_d_new_line = "Reason: " + p_d_new_line
                new_line += '\n[Verification Reasoning]\n'
                new_line += p_d_new_line
                try:
                    line_to_pred[int(id_)].append(new_line)
                except:
                    line_to_pred[int(id_)] = [new_line]
            else:
                tool_answer = data_point['tool_call_process'].split('Your Answer)')[-1].strip()
                divide_result = gemini_inference(model, model_name, divide_system, tool_answer, mandatory = "Single Fact: ", retry_num=5)
                single_fact_divided = divide_result.split('Single Fact:')
                for single_fact_ in single_fact_divided:
                    if len(single_fact_.strip()) < 10:
                        continue
                    new_line_copied = str(new_line.split(', Related information')[0])
                    new_line_copied += '\n[Verification Reasoning]\nSingle Fact:'
                    new_line_copied += single_fact_
                    try:
                        line_to_pred[int(id_)].append(new_line_copied.strip())
                    except:
                        line_to_pred[int(id_)] = [new_line_copied.strip()]

            print('[Verification Reasoning]')
            print(data_point['tool_call_process'].split('Your Answer)')[-1].strip())
            print()
            

    return pred_result_name, pred_json, skipped_line_to_pred, line_to_pred


def labeled_res(pred_result_name, dataver):
    if 'discharge' in pred_result_name:
        note_type = 'discharge'
    elif 'physician' in pred_result_name:
        note_type = 'physician'
    elif 'nursing' in pred_result_name:
        note_type = 'nursing'

    if 'val' in pred_result_name:
        test_val = 'val'
        test_val_dir = 'valid'
    elif 'test' in pred_result_name:
        test_val = 'test'
        test_val_dir = 'test'

    note_data = pd.read_csv(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/{note_type}_{test_val}.csv')
    idx = pred_result_name.split('_')[2].split('.json')[0]

    hadmid = note_data['HADM_ID'][int(idx)]
    try:
        with open(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/EHRReasonCon_{hadmid}_data.json', 'r', encoding='utf-8') as f:
            label_data = json.load(f)
    except:
        with open(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/EHRReasonCon_{hadmid}.0_data.json', 'r', encoding='utf-8') as f:
            label_data = json.load(f)

    data_list = label_data[list(label_data)[0]]

    sorted_data = sorted(data_list,key=lambda x: int(list(x.values())[0]['position']))

    data_line_to_labeled = {}

    for dat in sorted_data:
        for skip_key in ['all_queries', 'tag', 'almost_consistent', 'consistency_check_path', 'medical_knowledge_source', 'commonsense_medical_none', 'entity_type', 'source', 'evidence_row_ids']:
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
        print(list(dat)[0], dat[list(dat)[0]]['position'], dat[list(dat)[0]])

        try:
            data_line_to_labeled[int(dat[list(dat)[0]]['position'])].append(new_line.strip())
        except:
            data_line_to_labeled[int(dat[list(dat)[0]]['position'])] = [new_line.strip()]

    return data_line_to_labeled


def gemini_inference(model, model_name, system_prompt, prompt, mandatory = "", retry_num=5):
    try:
        import pickle
        with open('./eval_tokens.pkl', 'rb') as f:
            token_dict = pickle.load(f)
    except:
        token_dict = {'pro': {'input': 0, 'output': 0}, 'flash': {'input':0,'output':0}}
    for i in range(retry_num):
        try:
            if 'gemini-2.5-flash' == model_name:
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
            elif 'gemini-2.5-pro' in model_name:
                gemini_response = model.models.generate_content(model="gemini-2.5-pro", 
                                                                contents=prompt,
                                                                config=types.GenerateContentConfig(
                                                                    system_instruction=system_prompt,
                                                                    thinking_config=types.ThinkingConfig(thinking_budget=512)))
            elif 'gemini-2.0' in model_name:
                gemini_response = model.models.generate_content(model="gemini-2.0-flash", 
                                                                contents=prompt,
                                                                config=types.GenerateContentConfig(
                                                                    system_instruction=system_prompt))
            
            if mandatory.lower() in gemini_response.text.lower():
                break
        except:
            time.sleep(30)
    time.sleep(5)
    if 'pro' in model_name:
        token_dict['pro']['input'] += len(prompt+system_prompt)
        token_dict['pro']['output'] += len(gemini_response.text)
    else:
        token_dict['flash']['input'] += len(prompt+system_prompt)
        token_dict['flash']['output'] += len(gemini_response.text)
    with open('./eval_tokens.pkl', 'wb') as f:
        pickle.dump(token_dict, f)
    return gemini_response.text


def recall_res(preds, labeled, skipped, model, model_name, eval_mode):
    past_histories = deque(maxlen=5)
    with open(f"./prompt/evaluation_recall_system_{eval_mode}.txt", "r", encoding="utf-8") as f:
        recall_system = f.read()

    history_data_res = {"history":[], "data":[], "res":[]}
    
    tot_res = {"cor": 0 , 'incor': 0}
    for id_ in list(labeled):
        labeled_datas = labeled[id_]

        for pred_i in range(id_):
            if pred_i in list(preds):
                for p_pred in preds[pred_i]:
                    if p_pred not in past_histories:
                        past_histories.append(p_pred)
        
        preds_datas = []
        past_preds_datas = []
        future_preds_datas = []
        skipped_preds_datas = []

        tot_past_preds = past_histories.copy()
        if id_-1 in list(preds):
            if len(preds[id_-1]) > 5:
                tot_past_preds = preds[id_-1]
        
        if id_ in list(preds):
            preds_datas = preds[id_]
        
        if id_+1 in list(preds):
            future_preds_datas = preds[id_+1]
        if id_ in list(skipped):
            skipped_preds_datas = skipped[id_]

        history = "<Consistency check result>\n"
        history += "\n\n".join(list(tot_past_preds)+preds_datas+future_preds_datas)
        if len(skipped_preds_datas) > 0:
            history += "\n\n<Consistency check was already completed>\n" + "\n\n".join(skipped_preds_datas)

        for data in labeled_datas:
            print(history)
            print()
            print(id_, data)
            prompt = f"<Ground Truth Data>\nLine number: {id_}, {data}\n\n<Prediction Results>\n{history}"
            if 'gemini' in model_name:
                llm_res = gemini_inference(model, model_name, recall_system, prompt, mandatory = "Result: ", retry_num=5)
            elif 'gpt' in model_name:
                llm_res = get_azure_response(model, prompt, model_name, recall_system)
            print('<RECALL>')
            if "incorr" in llm_res.lower().split('result:')[1]:
                tot_res['incor'] += 1
                print('Incorrect')
            else:
                tot_res['cor'] += 1
                print('Correct')
            history_data_res['history'].append(history)
            history_data_res['data'].append(data)
            history_data_res['res'].append(llm_res)
    with open("./judge_res_3.pkl", "wb") as f:
        pickle.dump(history_data_res, f)
    return tot_res


def precision_res(preds, labeled, model, model_name, eval_mode):
    with open(f"./prompt/evaluation_precision_system_{eval_mode}.txt", "r", encoding="utf-8") as f:
        precision_system = f.read()
    tot_res = {"cor": 0 , 'incor': 0}
    
    for id_ in list(preds):
        preds_datas = preds[id_]
        
        labeled_datas = []
        past_past_labeled_datas = []
        past_labeled_datas = []
        future_labeled_datas = []
        future_future_labeled_datas = []
        if id_ in list(labeled):
            labeled_datas = labeled[id_]
        if id_-2 in list(labeled):
            past_past_labeled_datas = labeled[id_-2]
        if id_-1 in list(labeled):
            past_labeled_datas = labeled[id_-1]
        if id_+1 in list(labeled):
            future_labeled_datas = labeled[id_+1]
        if id_+2 in list(labeled):
            future_future_labeled_datas = labeled[id_+2]

        history = "\n\n".join(past_past_labeled_datas+past_labeled_datas+labeled_datas+future_labeled_datas+future_future_labeled_datas)

        for data in preds_datas:
            print(id_, data)
            print(history)
            if len(history.strip()) == 0:
                tot_res['incor'] += 1
                continue
            prompt = f"<Prediction Result>\nLine number: {id_}, {data}\n\n<Ground Truth Data>\n{history}"
            if 'gemini' in model_name:
                llm_res = gemini_inference(model, model_name, precision_system, prompt, mandatory = "Result: ", retry_num=5)
            elif 'gpt' in model_name:
                llm_res = get_azure_response(model, prompt, model_name, precision_system)
            print('<PRECISION>')
            if "incorr" in llm_res.lower().split('result:')[1]:
                tot_res['incor'] += 1
                print('Incorrect')
            else:
                tot_res['cor'] += 1
                print('Correct')
            
    return tot_res

if __name__ == "__main__":

    EVAL_MODE = os.getenv("EVAL_MODE", "harsh")  # {"harsh", "lenient"}
    _nt = os.getenv("NOTE_TYPE", "discharge"); _tv = os.getenv("TEST_VAL", "val")
    _nm = os.getenv("NER_METHOD", "cluster"); _dv = os.getenv("DATA_VERSION", "v2")

    # Which notes to score: NOTE_IDX for a single note, otherwise every note in the split
    # (same convention as Stage 1/2). Notes with no prediction JSON are skipped below.
    _idx = os.getenv("NOTE_IDX")
    if _idx is not None:
        note_ids = [int(_idx)]
    else:
        _dir = 'valid' if _tv == 'val' else 'test'
        note_ids = range(len(pd.read_csv(f'{EHR_REASONCON_DIR}/{_nt}/{_dir}/{_nt}_{_tv}.csv')))

    for i in note_ids:
        # Prediction JSON written by Stage 2 (tool_calling.py):
        #   concheck_result_{id}_{note_type}_{val|test}_{ner_method}_{data_version}_2.json
        pred_result_name = f"./result/concheck_result_{str(i)}_{_nt}_{_tv}_{_nm}_{_dv}_2.json"
        dataver = 'v2'
        vertex_model = init_vertex_model()
        try:
            pred_result_name, pred_json, skipped_line_to_pred, line_to_pred = organized_preds(pred_result_name, vertex_model, 'gemini-2.5-flash')
            data_line_to_labeled = labeled_res(pred_result_name, dataver)
        except Exception as e:
            print(f'[skip] note {i}: no usable prediction/label files ({type(e).__name__}: {e})')
            continue

        # gemini inference
        try:
            recall_ = recall_res(line_to_pred, data_line_to_labeled, skipped_line_to_pred, vertex_model, 'gemini-2.5-pro', EVAL_MODE)
            recall_value = recall_['cor'] / (recall_['cor']+recall_['incor'])
            recall_value = round(recall_value*100, 2)

            precision_ = precision_res(line_to_pred, data_line_to_labeled, vertex_model, 'gemini-2.5-pro', EVAL_MODE)
            precision_value = precision_['cor'] / (precision_['cor']+precision_['incor'])
            precision_value = round(precision_value*100, 2)
            print(pred_result_name)
            print('RECALL:', recall_value, recall_)
            print('PRECISION:', precision_value, precision_)
        except Exception as e:
            print(f'[error] judging note {i} failed ({type(e).__name__}: {e})')
            print('RECALL:', 0)
            print('PRECISION:', 0)