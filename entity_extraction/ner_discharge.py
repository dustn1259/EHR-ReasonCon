import pandas as pd
import argparse
import pandas as pd
import os
import json
import ast
import string
import ast
from utils.utils import *
from utils.post_process import *
from utils.ner_prompt_generator import *
from models import *


from utils.bm25_search_function import find_similar_records
from dotenv import load_dotenv
from google.genai import types
from utils.time_post_process import *
from datetime import datetime, timedelta
import pickle
import re
from collections import OrderedDict
import time


# EHR-ReasonCon dataset root (the PhysioNet bundle). Defaults to <release>/EHR_ReasonCon;
# override with the EHR_REASONCON_DIR env var (see check_dataset.py / README).
EHR_REASONCON_DIR = os.getenv(
    "EHR_REASONCON_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EHR_ReasonCon")),
).rstrip("/")


def gemini_inference(vertex_model, model_name, input):
    for i in range(5):
        try:
            if model_name == 'gemini-2.5-flash':
                return vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                contents=input, 
                                                config=types.GenerateContentConfig(
                                                        thinking_config=types.ThinkingConfig(thinking_budget=256)
                                                        ),).text
            elif model_name == "gemini-3-flash-preview":
                return vertex_model.models.generate_content(model="gemini-3-flash-preview", 
                                                contents=input, 
                                                config=types.GenerateContentConfig(
                                                        thinking_config=types.ThinkingConfig(thinking_level="low")
                                                        ),).text
            elif model_name == 'gemini-2.0-flash':
                return vertex_model.models.generate_content(model="gemini-2.0-flash", 
                                                contents=input).text
            elif model_name == 'gemini-2.5-pro':
                return vertex_model.models.generate_content(model="gemini-2.5-pro", 
                                                contents=input, 
                                                config=types.GenerateContentConfig(
                                                        thinking_config=types.ThinkingConfig(thinking_budget=256)
                                                        ),).text
        except:
            time.sleep(10)
            continue

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def extract_entities_updated(input_text):
  output_list = []
  lines = input_text.strip().split('\n')

  for line in lines:
    parts = line.split(':', 1)
    prefix = parts[0].strip()
    suffix = parts[1].strip() if len(parts) > 1 else ""

    # Conditions 1 & 3: when there is content before ':'
    if prefix:
      content_inside_brackets = suffix.strip()[1:-1]
      
      # 🆕 added rule: check whether the content contains '/'
      if '/' in content_inside_brackets:
        # Use a regular expression to find all numeric (including decimal) values.
        
        values = content_inside_brackets.split('/')
        # Add prefix to the list as many times as the number of values.
        count = len(values)
        if count > 0:
            items_to_add = [prefix]
            items_to_add.extend([f'{prefix}_{i}' for i in range(2, count + 1)])
            output_list.extend(items_to_add)
      else:
        # Existing rule: if there is no '/', add prefix only once.
        output_list.append(prefix)

    # Condition 2: when there is no content before ':' but there is content after it
    elif not prefix and suffix != '[]':
      content_inside_brackets = suffix.strip()[1:-1]
      if content_inside_brackets:
        items = [item.strip() for item in content_inside_brackets.split(',')]
        output_list.extend(items)

  return output_list

def merge_intersections(intervals):
    if not intervals:
        return []

    # 1. Sort by start point
    intervals.sort()
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        # Intersection check: intersects if current's start is <= last's end
        if current[0] <= last[1]:
            # Merge: extend the end point to the larger one
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged

def remove_supersets(intervals):
    result = []
    for i, (s1, e1) in enumerate(intervals):
        is_contained = False
        for j, (s2, e2) in enumerate(intervals):
            if i != j and s2 <= s1 and e1 <= e2:
                is_contained = True
                break
        if not is_contained:
            result.append([s1, e1])
    return result

def process_intervals(intervals):
    merged = merge_intersections(intervals)
    cleaned = remove_supersets(merged)
    return cleaned


def is_number(s):
  """Returns True if the string can be converted to an integer or float."""
  try:
    float(s)
    return True
  except ValueError:
    return False

def process_list_short(input_list):
  """Processes the list concisely using regular expressions. (regardless of the number of slashes)"""
  new_list = []
  for item in input_list:
    # If item has no slash, add it as is
    if '/' not in item:
      new_list.append(item)
      continue
    
    # Extract 'prefix', 'slash part', and 'suffix' at once with the revised regex
    match = re.search(r'(.*?)(\S+(?:/\S+)+)(.*)', item)
    
    if match:
      prefix, slash_part, suffix = match.groups()
      # Attach prefix/suffix to the split parts and add them to the list
      new_list.extend([f"{prefix}{part}{suffix}" for part in slash_part.split('/')])
    else:
      # If there is an exceptional case that does not match the regex, add the original as is
      new_list.append(item)
      
  return new_list

def parse_text_to_lines(raw_text: str) -> list[tuple[int, str]]:
    parsed_lines = []
    for line in raw_text.strip().split('\n'):
        # Use a regular expression to parse lines of the form 'number: content'.
        match = re.match(r'\s*(\d+):\s*(.*)', line)
        if match:
            line_num = int(match.group(1))
            content = match.group(2).strip()
            # Add only lines that have content.
            if content:
                parsed_lines.append((line_num, content))
    return parsed_lines

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', '', text).lower()


def extract_numbered_lines(text: str) -> list[tuple[int, str]]:
    """Extracts the text into a list of (line number, content)."""
    lines = []
    for line in text.strip().split('\n'):
        match = re.match(r'(\d+):\s*(.*)', line)
        if match:
            lines.append((int(match.group(1)), match.group(2).strip()))
    return lines

def standardize_text(text: str) -> str:
    """Standardizes the text for comparison (lowercase, strip surrounding whitespace)."""
    return text.lower().strip()

#--- Main function of the improved logic (renamed) ---
def create_line_mapping(subtext_check: str, output_check_number: str) -> dict[int, list[int]]:
    # 1) Parse
    sub_lines = extract_numbered_lines(subtext_check)
    out_lines = extract_numbered_lines(output_check_number)

    # 2) Expand (standardize) subtext + record cumulative length of line boundaries
    #    Put a single space between each line so they naturally join together
    std_sub_texts = [standardize_text(t) for _, t in sub_lines]
    sep = " "  # separator for joining lines
    pieces = []
    line_start_pos = []  # offset where each subline starts in the expanded string
    pos = 0
    for idx, txt in enumerate(std_sub_texts):
        line_start_pos.append(pos)
        pieces.append(txt)
        pos += len(txt)
        if idx != len(std_sub_texts) - 1:
            pieces.append(sep)
            pos += len(sep)
    flat_sub = "".join(pieces)

    # 3) To map offsets in the expanded string -> subtext line numbers,
    #    remember the [start, end) span of each line
    spans = []
    for i, (lnum, _) in enumerate(sub_lines):
        start = line_start_pos[i]
        end = start + len(std_sub_texts[i])
        spans.append((start, end, lnum))

    def charpos_to_subline_nums(start_pos: int, end_pos: int) -> list[int]:
        """Return the list of subtext line numbers that flat_sub's [start_pos, end_pos) span touches"""
        touched = []
        for s, e, lnum in spans:
            if not (end_pos <= s or e <= start_pos):  # if the spans overlap
                touched.append(lnum)
        return sorted(set(touched))

    # 4) Search each Output line in flat_sub in order (overlap allowed)
    mapping: dict[int, list[int]] = {}
    search_from = 0  # overlap allowed: keep searching from after the previous match's 'start'
    for out_num, out_text in out_lines:
        needle = standardize_text(out_text)
        if not needle:
            continue
        # Find (overlap allowed: search_from is at least the previous match's start position)
        found_at = flat_sub.find(needle, search_from)
        if found_at == -1:
            # If not found, shift by one and retry (rare, but defensive)
            found_at = flat_sub.find(needle, max(0, search_from - 1))
        if found_at != -1:
            start = found_at
            end = found_at + len(needle)
            mapping[out_num] = charpos_to_subline_nums(start, end)
            # Overlap-allowed point: set the next search start to 'this match's start position'
            # (the same line can contribute to multiple Outputs)
            search_from = found_at
        else:
            # If unmatched, just continue
            continue

    return mapping


def find_ner_locations(
    final_ner: List[str],
    subtext_check: str,
    output_check_number: str,
    mapping: Dict[int, List[int]]
) -> List[Dict[str, int]]:

    entity_locations = []
    last_real_location = None

    output_map  = dict(parse_text_to_lines(output_check_number))
    subtext_map = dict(parse_text_to_lines(subtext_check))

    search_line_num = -1
    search_char_idx = 0

    def make_pattern(token: str) -> re.Pattern:
        return re.compile(rf'(?<!\w){re.escape(token)}(?!\w)', re.I)

    for entity in final_ner:

        # 1) For a '_number' suffix, use the previous location
        if re.search(r'_\d+$', entity):
            entity_locations.append({entity: last_real_location or -1})
            continue

        regex = make_pattern(entity)
        found = False

        # ---------- 2) Search the output area ----------
        for out_line_num in sorted(output_map):
            if out_line_num < search_line_num:
                continue

            line = output_map[out_line_num]
            start_pos = search_char_idx if out_line_num == search_line_num else 0

            m = regex.search(line[start_pos:])
            if not m:
                continue

            # ---------- 3) output ↔ subtext mapping ----------
            subtext_lines = mapping.get(out_line_num)

            if subtext_lines:           # if there is a mapping, use it as is
                final_sub = (
                    subtext_lines[0] if len(subtext_lines) == 1
                    else next(
                        (n for n in subtext_lines
                         if re.search(rf'(?<!\w){re.escape(entity.split()[0])}(?!\w)',
                                      subtext_map.get(n, ''), re.I)),
                        subtext_lines[0]
                    )
                )
            else:                       # if there is no mapping, search the note directly
                start_sub = last_real_location if last_real_location is not None else -1
                final_sub = -1
                for sub_ln in sorted(subtext_map):
                    if sub_ln < start_sub:          # examine from the same line onward
                        continue
                    if regex.search(subtext_map[sub_ln]):
                        final_sub = sub_ln
                        break
                if final_sub == -1:
                    final_sub = out_line_num        # if still not found, fallback

            # ---------- 4) Update result and cursor ----------
            entity_locations.append({entity: final_sub})
            last_real_location = final_sub
            search_line_num = out_line_num
            search_char_idx = m.end() + start_pos if out_line_num == search_line_num else 0
            found = True
            break

        if not found:
            entity_locations.append({entity: -1})
    return entity_locations

def process_data_final(original_list, data_string):
    """
    Processes data ensuring all output values are wrapped in a list.
    """
    
    # 1. Parse the data_string into a dictionary (Unchanged)
    parsed_data = {}
    for line in data_string.strip().split('\n'):
        if ':' in line:
            key, value_str = line.split(':', 1)
            key = key.strip()
            if not key:
                continue
            value_str = value_str.strip()
            if value_str.startswith('[') and value_str.endswith(']'):
                content = value_str[1:-1].strip()
                values = [v.strip() for v in content.split(',')] if content else []
                parsed_data[key] = values

    # 2. Group original keys by their master key (Unchanged)
    grouped_keys = OrderedDict()
    for item in original_list:
        original_key = item[0]
        master_key = re.split(r'_\d+$', original_key)[0]
        if master_key not in grouped_keys:
            grouped_keys[master_key] = []
        grouped_keys[master_key].append(original_key)
        
    # 3. Determine the final value for each key
    final_values = {}
    for master_key, related_keys in grouped_keys.items():
        master_value_list = parsed_data.get(master_key)
        
        # Splitting logic for '/'
        if (len(related_keys) > 1 and 
            master_value_list and 
            len(master_value_list) == 1 and 
            '/' in master_value_list[0]):
            
            master_string = master_value_list[0]
            parts = [p.strip() for p in master_string.split('/') if p.strip()]

            if len(parts) == len(related_keys):
                first_part = parts[0]
                last_space_index = first_part.rfind(' ')
                base = first_part[:last_space_index + 1] if last_space_index != -1 else ''
                
                distributed_values = []
                for p in parts:
                    variable_part = p[p.rfind(' ') + 1:] if p.rfind(' ') != -1 else p
                    distributed_values.append(base + variable_part)
                
                for key, val in zip(related_keys, distributed_values):
                    # --- MODIFICATION 1 ---
                    # Always wrap the final distributed value in a list.
                    final_values[key] = [val] 
                continue

        # --- MODIFICATION 2 ---
        # Default logic: Remove the condition that unwraps single-element lists.
        # This ensures the value remains a list.
        value_to_assign = [] 
        if master_value_list:
            value_to_assign = master_value_list # Directly assign the list from parsed_data
        
        for key in related_keys:
            final_values[key] = value_to_assign

    # 4. Assemble the final list (Unchanged)
    output_list = []
    for item in original_list:
        key = item[0]
        output_list.append([key, item[1], final_values.get(key)])
        
    return output_list


def scope_question_to_dict(text_data: str) -> Dict[int, str]:

    parsed_dict = {}

    # --- Revised regular expression ---
    # (.*?): non-greedily find the text after 'Answer'.
    # (?=\s*\d+\.|\Z): set the boundary so it only matches up to just before the 'next number' or 'end of string'.
    pattern = re.compile(r"(\d+)\..*?Answer:?\s*(.*?)(?=\s*\d+\.|\Z)", re.DOTALL | re.IGNORECASE)

    # Use findall to find all (key, value) pairs across the entire text at once.
    matches = pattern.findall(text_data)

    for key, value in matches:
        # Add the found result to the dictionary.
        parsed_dict[int(key)] = value.strip()
        
    return parsed_dict

def parse_gemini_output(input_str: str) -> List[list]:
    # Strip leading/trailing whitespace from the input string; if it does not start with "Line number", return an empty list.
    if not input_str.strip().startswith("Line number"):
        return []

    # List to store the results
    results = []

    # Define the regular expression pattern
    # The meaning of each group is as follows:
    # 1. (\d+): line number (one or more digits)
    # 2. (.*?): entity (all characters from after '. ' up to before ' - [', matched minimally)
    # 3. (.*?): ehr entity (all characters between '[' and ']')
    # 4. (.*?): associated value (all characters between the second '[' and ']')
    pattern = re.compile(r"Line number (\d+)\. (.*?) - \[(.*?)\] - \[(.*?)\]")
    # Process the input string line by line
    for line in input_str.strip().splitlines():
        match = pattern.match(line.strip())
        
        if match:
            # Extract each piece of data from the matched groups and strip whitespace.
            line_number = match.group(1).strip()
            entity = match.group(2).strip()
            ehr_entity_str = match.group(3).strip()
            associated_value = match.group(4).strip()

            # If ehr_entity is not empty, put it in a list; if empty, make it an empty list.
            ehr_entity_list = [ehr_entity_str] if ehr_entity_str else []

            # Assemble the list according to the final result format.
            results.append([entity, line_number, associated_value, ehr_entity_list])
    return results

def process_cluster_output(text: str) -> list:
    pattern = re.compile(r"^Line number\s*(\d+)\.\s*(.*?):\s*\[(.*?)]\s*-\s*\[(.*?)]$", re.MULTILINE)

    matches = pattern.findall(text)

    processed_list = []
    for match in matches:
        line_num = match[0]
        entity = match[1].strip()
        value = match[2].strip()
        
        # The linking entity list is split by commas (,) into a list.
        linking_entities_str = match[3].strip()
        linking_entities = [e.strip() for e in linking_entities_str.split(',')]

        processed_list.append([entity, line_num, value, linking_entities])

    return processed_list

def extract_data_info_final(data):
    """
    Data extraction function with the final logic applied.
    - First filters out the excluded tables, then handles special cases.
    - For all remaining tables, looks for 'valuenum' first, and 'value' if not present.
    """
    result_list = []
    
    # Iterate over all patient records in the data dictionary
    for patient_records in data.values():
        for record in patient_records:
            for data_key, details in record.items():
                
                # --- Exclusion logic ---
                table_name = details.get('table')
                should_skip = False
                
                # Exclude based on the 'table' key
                if table_name in ['d_icd_diagnoses', 'd_icd_procedures']:
                    should_skip = True

                # Exclude based on keys inside 'data' (in case the table key is missing)
                if not should_skip and details.get('data'):
                    keys = details['data'][0].keys()
                    if 'd_icd_diagnoses' in keys or 'd_icd_procedures' in keys:
                        should_skip = True
                
                if should_skip:
                    continue
                # --- End of exclusion logic ---

                position = details.get('position')
                value = []  # default value
                data_list = details.get('data', [])

                if data_list:
                    data_item = data_list[0]
                    
                    # --- Per-table value extraction logic ---
                    # 1. Handle special cases first
                    if table_name == 'microbiologyevents':
                        value = []
                    elif table_name == 'prescriptions':
                        extracted_val = data_item.get('prescriptions', {}).get('dose_val_rx')
                        if extracted_val is not None:
                            value = extracted_val
                    # 2. All other general cases
                    else:
                        for sub_dict in data_item.values():
                            if isinstance(sub_dict, dict):
                                # 2-1. Look for 'valuenum' first
                                if 'valuenum' in sub_dict:
                                    value = sub_dict.get('valuenum', [])
                                    break
                                # 2-2. If 'valuenum' is absent, look for 'value'
                                elif 'value' in sub_dict:
                                    value = sub_dict.get('value', [])
                                    break
                
                result_list.append([data_key, position, value])
                
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Process a clinical note based on YAML config.")
    parser.add_argument("config_path", help="Path to the YAML configuration file")
    args = parser.parse_args()
    config = load_config(args.config_path)

    note_type = "discharge"                    # fixed per script
    testval = os.getenv("TEST_VAL", "val")     # {"val", "test"}
    test_val_dir = 'valid' if testval == 'val' else testval
    model_name_file = 'gemini'

    df = pd.read_csv(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/{note_type}_{testval}.csv')
    vertex_model = init_vertex_model()
    azure_client = init_azure_client()
    deployment = os.getenv("AZURE_DEPLOYMENT")
    NOTE_IDX = os.getenv("NOTE_IDX")  # e.g. NOTE_IDX=6 to run one note; unset = run all
    for idx, note in enumerate(df["TEXT"]):
        note= note.replace("__","")
        if NOTE_IDX is not None and idx != int(NOTE_IDX):
            continue
        print("idx", idx)
        adm_time=df['ADMITTIME'][idx]
        chart_time=df['CHARTDATE'][idx]
        hadmid = df['HADM_ID'][idx]
        with open("../tool_calling/current_hadm_id.txt", "w", encoding="utf-8") as f:
            f.write(f"{note_type}, {testval}, {hadmid}")

        target_hadm_id = hadmid.astype(int)
        ## df for data that exists in the db for the actual patient
        record_data_df = pd.read_csv(f'{EHR_REASONCON_DIR}/patient_data/{target_hadm_id}/patient_records.csv')
        record_data_df['TOP_VALUES'] = record_data_df['TOP_VALUES'].str[:-1] + ", 'NaN']"
        record_str_list = [f"{label}: {values}" for label, values in zip(record_data_df['LABEL'], record_data_df['TOP_VALUES'])]
        record_str_list = "\n".join(record_str_list)
        ## df for data that does not exist in the db for the actual patient (added)
        remain_data_df = pd.read_csv(f'{EHR_REASONCON_DIR}/patient_data/{target_hadm_id}/catalog_items.csv')
        remain_data_df['TOP_VALUES'] = remain_data_df['TOP_VALUES'].str[:-1] + ", 'NaN']"

        numbered_note = add_line_numbers(note)
        subtext_prompt = open_file("./prompt/ner_v2/extract_subtext.txt").replace("<<<Clinical_Note>>>", numbered_note)
        gemini_response = gemini_inference(vertex_model, "gemini-2.5-flash", subtext_prompt)
        print(gemini_response)

        ner_time_result = []

        sub_old=extract_subtext(numbered_note, gemini_response) 
        sub = {}
        new_key_index = 0
        sorted_keys = sorted(sub_old.keys(), key=lambda x: int(x.split('_')[1]))

        time_prompt = open_file(config["prompts"]["time_summary"]).replace("<<<CLINICAL_NOTE>>>", note)  #**need to fix the confusing code around the time_Summary and time_sum parts!
        time_summary = get_azure_response(azure_client, time_prompt, deployment)

        for key in sorted_keys:
            subtext_ = sub_old[key]

            # Group the keywords that require text splitting with 'or' and handle them at once.
            if 'discharge medication' in subtext_.lower() or 'pertinent result' in subtext_.lower() or 'laboratory data' in subtext_.lower():
                
                # Select a different regex pattern depending on which keyword is included.
                if 'discharge medication' in subtext_.lower():
                    pattern = r'\n\d+:\t?\.?\n(?!\s*\d+:\s*\d+\.)'
                else: # case of 'pertinent result' or 'laboratory data'
                    pattern = r"\n\d+:\s*\.?\s*\n(?!\s*\d+:\s*\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]\s*\d{2}:\d{2}[AP]M)"
                
                splitted_out = []
                if len(re.findall(pattern, subtext_)) > 0:
                    splitted_out = re.split(pattern, subtext_)
                else:
                    # If there is no part matching the pattern, use the original text as is.
                    splitted_out = [subtext_]
                
                ## < BOLD >The saving logic now applies to both cases.</BOLD >
                for part in splitted_out:
                    # Add it, excluding cases where only whitespace remains after splitting.
                    if part and part.strip():
                        sub[f'subject_{new_key_index}'] = part.strip()
                        new_key_index += 1
            else:
                # If the condition does not apply, add it as is like before.
                sub[f'subject_{new_key_index}'] = subtext_
                new_key_index += 1
        
        with open(f'./output/subtext_divided_{note_type}_{testval}_{idx}_{model_name_file}_2.pkl', 'wb') as f:
            pickle.dump(sub, f)

        for i in sub:
            print("NOW ON :", i)
            subtext_ = sub[i]

            start_text_line, end_text_line = [int(re.match(r'\s*(\d+):', l).group(1)) for l in (subtext_.splitlines()[0], subtext_.splitlines()[-1])]

            print(start_text_line, end_text_line)
            subtext_check = subtext_

            subtext_ = re.sub(r"^\d+:\s*", "", subtext_, flags=re.MULTILINE)
            reformat_prompt=open_file('./prompt/ner_v2/reformat_prompt.txt')
            reformat_prompt=reformat_prompt.replace("<<<DISCHARGE_SUMMARY>>>",subtext_)
            output = gemini_inference(vertex_model, "gemini-2.5-flash", reformat_prompt)
            output_check_number=add_line_numbers(output)
            temp_dict_line_num = create_line_mapping(subtext_check, output_check_number)

            print(i)
            if ('admission date' in subtext_.lower() and
                'discharge date' in subtext_.lower()):
                continue
            final_ner_output=[] ##this is where the final result should ultimately be filled in
            ## case already stored in the patient's ehr
            prompt_cluster_record_prompt=open_file('./prompt/cluster/record_cluster.txt').replace("<<<PREDEFINED_VALUES>>>",record_str_list).replace("<<<CLINICAL_NOTE>>>",subtext_check)
            gemini_record_cluster = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_record_prompt)
            final_ner_output1=process_cluster_output(gemini_record_cluster)  ##entity extraction result based on the content stored in the patient's ehr [[entity, linenumber, value, [litem1, item2,... ]],[entity, linenumber, value, [litem1, item2,... ]]]
            str_final_ner_output1 = '\n'.join([str(item) for item in final_ner_output1]) #convert final_ner_output1 to str


            ## case not in the patient's ehr but only in mimic
            unique_groups =remain_data_df['GROUP'].unique()  ##first extract all groups
            alphabet = string.ascii_uppercase

            # Use enumerate to get the index and value together and combine into the desired format
            formatted_list = [f"({alphabet[i]}) {group}" for i, group in enumerate(unique_groups)] #A) Group1 B) Group2... build multiple-choice options from groups
            print("Let's start ehr table")
            # Join all items of the list with a space (" ") into a single string
            result_string = " ".join(formatted_list)
            prompt_cluster_remain_prompt=open_file('./prompt/cluster/cluster_group_quiz.txt').replace("<<<SELECTED_CATEGORY>>>",result_string).replace("<<<CLINICAL_NOTE>>>",subtext_check)
            gemini_remain_cluster = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_remain_prompt)
            flattened_remain_subgroup = []


            if gemini_remain_cluster.strip() != '[]':
                result_list_remain = [re.sub(r'^\s*\([A-Z]\)\s*', '', item).strip() for item in gemini_remain_cluster.strip('[]').split(',')] #turn the gemini_remain_cluster result into a list
                final_df_subgroup = remain_data_df[remain_data_df['GROUP'].isin(result_list_remain)] ##find GROUP from the extracted ones -> build a new df (made only of the extracted groups)
                unique_groups_subgroup=final_df_subgroup['SUBGROUP'].unique() ## uniquely extract subgroups based on the selected GROUP

                # Use enumerate to get the index and value together and combine into the desired format
                formatted_list_ = [f"({i+1}) {group}" for i, group in enumerate(unique_groups_subgroup)] # build options like 1)subgroup 2)subgroup
                result_string_subgroup = " ".join(formatted_list_)
                prompt_cluster_remain_prompt_subgroup=open_file('./prompt/cluster/cluster_group_quiz.txt').replace("<<<SELECTED_CATEGORY>>>",result_string_subgroup).replace("<<<CLINICAL_NOTE>>>",subtext_check)
                gemini_remain_cluster_subgroup = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_remain_prompt_subgroup)
                if gemini_remain_cluster_subgroup.strip() != '[]':
                    try:
                        subgroup_list_output=[s.split(')', 1)[1].strip() for s in gemini_remain_cluster_subgroup.strip('[]').split(', (')]  ##turn the selected subgroups into a list
                    except:
                        subgroup_list_output = []
                    subgroup_df = remain_data_df[remain_data_df['SUBGROUP'].isin(subgroup_list_output)] ##build a new df using only the selected subgroups
                    subgroup_df=subgroup_df.reset_index(drop=True)
                    subgroup_label_list = subgroup_df['LABEL'].astype(str).tolist() ## extract LABEL values from the newly subgroup-built df
                    chunk_size = 200
                    separator = ", "  # separator to place between items (e.g. ", " or " | ")
                    chunked_strings = [separator.join(subgroup_label_list[i:i + chunk_size]) for i in range(0, len(subgroup_label_list), chunk_size)] ##split into small lists of 200 each
                    print(len(chunked_strings))
                    final_remain_subgroup_total=[]
                    for c_id, chunk_ in enumerate(chunked_strings):
                        print("NOW ON2:  ",c_id)
                        final_remain_subgroup_prompt=open_file('./prompt/cluster/cluster_quiz1.txt').replace("<<<CLINICAL NOTE>>>",subtext_check).replace("<<<CHECKED LIST>>>",chunk_).replace("<<<LAST_EXTRACTED>>>",str_final_ner_output1)
                        final_remain_subgroup_output = gemini_inference(vertex_model, "gemini-2.5-flash", final_remain_subgroup_prompt)
                        final_remain_subgroup_list=parse_gemini_output(final_remain_subgroup_output)
                        final_remain_subgroup_list = [item for item in final_remain_subgroup_list if item[-1] != [''] and item[-1] != []]
                        final_remain_subgroup_total.append(final_remain_subgroup_list)
                        final_remain_subgroup_total = [item for item in final_remain_subgroup_total if item]
                    flattened_remain_subgroup = [item for sublist in final_remain_subgroup_total for item in sublist]
                else:
                    print("when the 2nd (subgroup selection) is an empty list")
                    ### needs to be written here
            else:
                print("when the 1st is an empty list, skip everything and merge")
            combined_list = flattened_remain_subgroup + final_ner_output1  
            final_ner_output_ = sorted(combined_list, key=lambda x: int(x[1])) ## linenumber sorting 
            final_ner_output = []
            for item in final_ner_output_:
                if item[:-1] not in final_ner_output:
                    final_ner_output.append(item[:-1])  ##remove the corresponding item part  [[entity, linenumber, value, [litem1, item2,... ]] -> [[[entity, linenumber, value],]
            

            ### from here, need to write code that merges org_note with linenumber
            parsed_data = parse_medical_record(subtext_check)
            formmated_ner_ouput = ""
            for line_i in list(parsed_data):
                formmated_ner_ouput += f"{line_i}:\n"
                line_text = ""
                for ner_pair in final_ner_output:
                    if int(ner_pair[1]) == int(line_i):
                        if len(ner_pair[2]) == 0:
                            line_text += " {"+ f"{ner_pair[0]}:" + "}"
                        else:
                            line_text += " {"+ f"{ner_pair[0]}:{ner_pair[2]}" + "}"
                
                if line_text == "":
                    line_text = " {:}"
                formmated_ner_ouput += f"output:{line_text}\n"
            formmated_ner_ouput = f"```\n{formmated_ner_ouput}```"
            result_parsing = combine_data(parsed_data, formmated_ner_ouput)
            #discharge diagnoses','major surgical or invasive procedure','discharge diagnosis','secondary diagnosis','primary diagnosis','\nsecondary\n
            keywords_scope = ['secondary','primary diagnosis','secondary diagnosis','discharge diagnosis','major surgical or invasive procedure','discharge diagnoses','LABORATORY DATA','physical examination','physical exam', 'laboratory/radiologic data', 'discharge medication', 'admission labs', 'chief complaint','discharge lab']
            result_parsing = [row + [any(keyword.lower() in subtext_.lower() for keyword in keywords_scope)] for row in result_parsing] #if true here, skip running scope  #added here
            if len(result_parsing)==0:
                continue
            else:
                time_question_prompt=open_file(config["prompts"]["time_adm_dis"]).replace('<<<TEMPORTAL_EXPRESSION>>>', time_summary)
                time_adm_dis = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt)
                if '<Answer>:' in time_adm_dis:
                    time_adm_dis = time_adm_dis.split('<Answer>:')[1].strip()
                elif '<Answer>' in time_adm_dis:
                    time_adm_dis = time_adm_dis.split('<Answer>')[1].strip()
                adm_period = time_adm_dis.split('Admission:')[1].split('Discharge')[0].strip()
                dis_period = time_adm_dis.split('Discharge:')[1].split('Whole Range')[0].strip()
                whole_range = time_adm_dis.split('Whole Range:')[1].strip()
                
                if ' to ' not in adm_period:
                    base_date = datetime.strptime(adm_time, '%Y-%m-%d %H:%M:%S').date()
                    adm_start = base_date - timedelta(days=1)
                    adm_start = adm_start.isoformat()
                    adm_end = base_date + timedelta(days=1)
                    adm_end = adm_end.isoformat()
                    adm_period = f"{adm_start} to {adm_end}"
                if ' to ' not in dis_period:
                    base_date = datetime.strptime(chart_time, '%Y-%m-%d').date()
                    dis_start = base_date - timedelta(days=1)
                    dis_start = dis_start.isoformat()
                    dis_end = base_date + timedelta(days=1)
                    dis_end = dis_end.isoformat()
                    dis_period = f"{dis_start} to {dis_end}"
                if ' to ' not in whole_range:
                    whole_start = adm_period.split(' to ')[0]
                    whole_end = dis_period.split(' to ')[1]
                    whole_range = f"{whole_start} to {whole_end}"

                for _r in result_parsing:
                    if not _r[-1]: #uncomment here later
                        print("items that need scope checking:")
                        print(_r[0],":",_r[3])
                        try:
                            adm_time_scope = adm_period.split(' to ')[0]
                            chart_time_scope= dis_period.split(' to ')[1]
                            scope_prompt=open_file(config["prompts"]["discharge_summary_scope"]).replace("<<<Admission>>>",adm_time_scope).replace("<<<Chartdate>>>",chart_time_scope).replace("<<<CLINICAL NOTE>>>",subtext_).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<LINE_NUMBER>>>",_r[0])
                            scope_output = gemini_inference(vertex_model, "gemini-2.5-flash", scope_prompt)
                            scope_output_=extract_text_before_colon(scope_output)
                            if 'yes' in scope_output_.lower():
                                continue
                        except:
                            print("ENTITY ERROR:   ", print(_r[3]))  
                    

                    # Explicitly mentioned "physical exam on admission", "discharge medication", "chief complaint"
                    if any(keyword in subtext_check.lower() for keyword in ['secondary','primary diagnosis','secondary diagnosis', 'discharge diagnosis','major surgical or invasive procedure','discharge diagnoses']):
                        final_time_result = whole_range
                        time_related_info = "whole range"
                        time_first_output = "diagnosis or procedure"
                        time_second_output = "diagnosis or procedure"                           
                    elif "physical exam on admission" in subtext_check.lower():
                        final_time_result = adm_period
                        time_related_info = "admission"
                        time_first_output = "physical exam on admission"
                        time_second_output = "physical exam on admission"
                    elif "physical examination on admission" in subtext_check.lower():
                        final_time_result = adm_period
                        time_related_info = "admission"
                        time_first_output = "physical exam on admission"
                        time_second_output = "physical exam on admission"
                    elif "physical examination on presentation" in subtext_check.lower():
                        final_time_result = adm_period
                        time_related_info = "admission"
                        time_first_output = "physical exam on admission"
                        time_second_output = "physical exam on admission"
                    elif "discharge medication" in subtext_check.lower():
                        final_time_result = dis_period
                        time_related_info = "discharge"
                        time_first_output = "discharge medication"
                        time_second_output = "discharge medication"
                    elif "chief complaint" in subtext_check.lower():
                        final_time_result = whole_range
                        time_related_info = "whole range"
                        time_first_output = "chief complaint"
                        time_second_output = "chief complaint"
                    elif "\tphysical exam\n" in subtext_check.lower():
                        final_time_result = whole_range
                        time_related_info = "whole range"
                        time_first_output = "physical exam"
                        time_second_output = "physical exam"
                    elif "\tphysical exam:\n" in subtext_check.lower():
                        final_time_result = whole_range
                        time_related_info = "whole range"
                        time_first_output = "physical exam"
                        time_second_output = "physical exam"
                    else:
                        time_related_info, time_first_output, time_output_first_text, final_time_result = first_time_extraction(config, subtext_check, _r, vertex_model, adm_period, dis_period, whole_range, time_summary)
                        time_second_output, time_output_text, final_time_result = second_time_processing(config, subtext_check, _r, vertex_model, time_summary, final_time_result)
                    
                        if final_time_result == '':
                            time_correction_prompt= 'Please complete the response based on the given time information.\nOutput format: {admission date in yyyy-mm-dd format} to {discharge date in yyyy-mm-dd format}\nNote: Do not include any other text besides the output format.\nTime Information)\n<<<TIME_SUMMARY>>>'
                            time_correction_prompt=time_correction_prompt.replace("<<<TIME_SUMMARY>>>",time_summary)
                            gemini_response = gemini_inference(vertex_model, "gemini-2.5-flash", time_correction_prompt)
                            final_time_result=gemini_response

                    _r.pop()
                    _r.append([final_time_result, time_related_info, time_first_output, time_second_output])
                    ner_time_result.append(_r)

        with open(f'./output/clusterner_time_result_{note_type}_{testval}_{idx}_v2_{model_name_file}_2.pkl', 'wb') as f:
            pickle.dump(ner_time_result, f)
                

if __name__ == "__main__":
    main()
 