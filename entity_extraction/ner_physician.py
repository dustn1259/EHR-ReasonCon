import pandas as pd
import argparse
import pandas as pd
import os
import ast
import ast
from utils.segmentation_utils import *
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
from typing import List, Dict, Pattern
import string
import time


# EHR-ReasonCon dataset root (the PhysioNet bundle). Defaults to <release>/EHR_ReasonCon;
# override with the EHR_REASONCON_DIR env var (see check_dataset.py / README).
EHR_REASONCON_DIR = os.getenv(
    "EHR_REASONCON_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EHR_ReasonCon")),
).rstrip("/")


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

      # 🆕 Added rule: check whether the content contains '/'
      if '/' in content_inside_brackets:
        # Use a regular expression to find all numeric (including decimal) values.

        # From the result split by split('/'), keep only the non-empty items in the values list.
        values = [value for value in content_inside_brackets.split('/') if value]
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
        # Overlap check: overlaps if current's start is at or before last's end
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
  """Returns True if the string can be converted to an integer or a float."""
  try:
    float(s)
    return True
  except ValueError:
    return False

def gemini_inference(vertex_model, model_name, input):
    for i in range(5):
        try:
            if model_name == 'gemini-2.5-flash':
                return vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                contents=input, 
                                                config=types.GenerateContentConfig(
                                                        thinking_config=types.ThinkingConfig(thinking_budget=256)
                                                        ),).text
            elif model_name == 'gemini-3-flash-preview':
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

def process_list_short(input_list):
  """Processes the list concisely using regular expressions. (Regardless of the number of slashes)"""
  new_list = []
  for item in input_list:
    if '/' not in item and '\n' not in item:
      new_list.append(item)
      continue
    
    match = re.search(r'(.*?)(\S+(?:[/\n]\S+)+)(.*)', item)
    
    if match:
      prefix, part_to_split, suffix = match.groups()
      
      # Add '+' to treat consecutive delimiters as one
      split_parts = re.split(r'[/]+', part_to_split)
      
      new_list.extend([f"{prefix}{part}{suffix}" for part in split_parts])
    else:
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
    """Extracts the text as a list of (line number, content)."""
    lines = []
    for line in text.strip().split('\n'):
        match = re.match(r'(\d+):\s*(.*)', line)
        if match:
            lines.append((int(match.group(1)), match.group(2).strip()))
    return lines

def standardize_text(text: str) -> str:
    """Standardizes the text for comparison (lowercase, trim leading/trailing whitespace)."""
    return text.lower().strip()

#--- Main function of the improved logic (renamed) ---
def create_line_mapping(subtext_check: str, output_check_number: str) -> dict[int, list[int]]:
    # 1) Parse
    sub_lines = extract_numbered_lines(subtext_check)
    out_lines = extract_numbered_lines(output_check_number)

    # 2) Expand (standardize) subtext + record cumulative length at line boundaries
    #    Put a single space between each line so they join naturally
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
    #    remember each line's [start, end) span
    spans = []
    for i, (lnum, _) in enumerate(sub_lines):
        start = line_start_pos[i]
        end = start + len(std_sub_texts[i])
        spans.append((start, end, lnum))

    def charpos_to_subline_nums(start_pos: int, end_pos: int) -> list[int]:
        """Return the list of subtext line numbers that the [start_pos, end_pos) span of flat_sub touches"""
        touched = []
        for s, e, lnum in spans:
            if not (end_pos <= s or e <= start_pos):  # if the spans overlap
                touched.append(lnum)
        return sorted(set(touched))

    # 4) Search each Output line in flat_sub in order (overlap allowed)
    mapping: dict[int, list[int]] = {}
    search_from = 0  # overlap allowed: keep searching after the previous match's 'start'
    for out_num, out_text in out_lines:
        needle = standardize_text(out_text)
        if not needle:
            continue
        # Search (overlap allowed: search_from is at or after the previous match's start position)
        found_at = flat_sub.find(needle, search_from)
        if found_at == -1:
            # If not found, retry shifted by one (rare, but a safeguard)
            found_at = flat_sub.find(needle, max(0, search_from - 1))
        if found_at != -1:
            start = found_at
            end = found_at + len(needle)
            mapping[out_num] = charpos_to_subline_nums(start, end)
            # Overlap-allowed point: set the next search start to 'this match's start position'
            # (the same line can contribute to multiple Outputs)
            search_from = found_at
        else:
            # On no match, just continue
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
        return re.compile(rf'(?<!\w){re.escape(token)}(?:\'s)?(?!\w)', re.I)

    for entity in final_ner:

        # 1) The '_number' suffix uses the previous location
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

            if subtext_lines:           # if a mapping exists, use it as is
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

def find_ner_locations_modified(
    final_ner: List[str],
    subtext_check: str,
    output_check_number: str,
    mapping: Dict[int, List[int]]
) -> List[Dict[str, int]]:
    """
    Finds the corresponding location (line number) in the text based on the list of NER entities.

    - When the same 'base entity name' appears consecutively, copies the location of the first entity.
    - The 'base entity name' is the name excluding the parentheses and their contents. (e.g. 'Wgt (current)' -> 'Wgt')
    - This 'base entity name' is also used when searching the text.
    """
    entity_locations = []
    last_real_location = None

    output_map = dict(parse_text_to_lines(output_check_number))
    subtext_map = dict(parse_text_to_lines(subtext_check))

    search_line_num = -1
    search_char_idx = 0

    def make_pattern_1(token: str) -> Pattern:
        """Creates a regex pattern that matches the whole word."""
        return re.compile(rf'(?<!\w){re.escape(token)}(?:\'s)?(?!\w)', re.I)

    for i, entity in enumerate(final_ner):
        # Remove the parentheses and their contents to extract the 'base entity name'. (e.g. 'Wgt (current)' -> 'Wgt')
        base_entity = re.sub(r'\s*\(.*\)', '', entity).strip()

        # If the 'base entity name' is the same as the previous entity, copy the location and move on.
        if i > 0:
            prev_base_entity = re.sub(r'\s*\(.*\)', '', final_ner[i-1]).strip()
            if base_entity == prev_base_entity and base_entity != '': # only when the base entity name is not empty
                previous_location = list(entity_locations[-1].values())[0]
                entity_locations.append({entity: previous_location})
                continue

        # If a '_number' suffix is attached, use the previous location.
        if re.search(r'_\d+$', entity):
            entity_locations.append({entity: last_real_location or -1})
            continue

        # Use the 'base entity name' when searching.
        regex = make_pattern_1(base_entity)
        found = False

        # --- 1. Search for the entity in the output area ---
        for out_line_num in sorted(output_map):
            if out_line_num < search_line_num:
                continue

            line = output_map[out_line_num]
            start_pos = search_char_idx if out_line_num == search_line_num else 0

            match = regex.search(line[start_pos:])
            if not match:
                continue

            # --- 2. Map the output location to the subtext location ---
            subtext_lines = mapping.get(out_line_num)
            final_sub_line = -1

            if subtext_lines:
                # If there are multiple mapped subtext lines, preferentially find the one containing the entity name
                final_sub_line = next(
                    (ln for ln in subtext_lines if regex.search(subtext_map.get(ln, ''))),
                    subtext_lines[0] # if not found, use the first mapped line
                )
            else:
                # If there is no mapping info, search directly across the entire subtext
                start_search_sub_line = last_real_location if last_real_location is not None else -1
                for sub_ln in sorted(subtext_map):
                    if sub_ln < start_search_sub_line:
                        continue
                    if regex.search(subtext_map[sub_ln]):
                        final_sub_line = sub_ln
                        break
                # If not found in subtext, use the output line number instead (fallback policy)
                if final_sub_line == -1:
                    final_sub_line = out_line_num

            # --- 3. Store the result and update the cursor for the next search ---
            entity_locations.append({entity: final_sub_line})
            last_real_location = final_sub_line
            search_line_num = out_line_num
            search_char_idx = match.end() + start_pos
            found = True
            break # search succeeded for the current entity, move on to the next entity

        if not found:
            entity_locations.append({entity: -1}) # when the entity is not found in the text

    return entity_locations

def process_data_final_revised(original_list, data_string):
    """
    Parses data and sequentially assigns each value to its corresponding key
    in the original list's order.
    """
    
    # 1. Parse the data_string, accumulating all values for each key (same as before)
    # e.g.: {'BP': [['97'], ['48'], ...], 'HR': [['83'], ['81'], ['109']]}
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
                
                if key not in parsed_data:
                    parsed_data[key] = []
                parsed_data[key].append(values)

    # --- Entire logic changed ---
    # 2. Build the final list, assigning values sequentially
    output_list = []
    # A counter that tracks which value index to fetch for each key ('HR', 'BP')
    value_counters = {key: 0 for key in parsed_data.keys()}

    for item in original_list:
        key = item[0]
        line_num = item[1]

        # Fetch the values to assign
        all_values = parsed_data.get(key)

        value_to_assign = [] # default value
        if all_values:
            # Fetch the counter corresponding to the current key
            current_index = value_counters.get(key, 0)

            # Check whether there are values left to assign
            if current_index < len(all_values):
                value_to_assign = all_values[current_index]
                # Increment this key's counter by 1 so the next value is fetched next time
                value_counters[key] += 1
        
        output_list.append([key, line_num, value_to_assign])
        
    return output_list


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


def extract_list_from_text(text_input: str) -> list:
    """
    Finds the "[ [...], [...] ]" pattern in the text and converts it to a Python list.
    Uses regular expressions to handle whitespace and line breaks flexibly.
    """
    try:
        # 1. Find the pattern '[', optional whitespace, '[', any characters (.), ']', optional whitespace, ']'.
        # The re.DOTALL option makes the '.' character also match line break characters (\n).
        match = re.search(r'\[\s*\[.*?\]\s*\]', text_input, re.DOTALL)

        # 2. Check whether there is a part matching the pattern.
        if match:
            # 3. Get the entire matched string ('[ [...] ]').
            list_string = match.group(0)

            # 4. Safely convert the obtained string into an actual Python list.
            final_list = ast.literal_eval(list_string)

            return final_list

    except (SyntaxError, ValueError):
        # When the string format is invalid and cannot be converted to a list
        return []

    # When no content matching the pattern is found
    return []


def extract_list_from_text_old(text_input: str) -> list:
    try:
        # 1. Split the text on "Final_Output:", then take the latter part ([1]).
        list_string = text_input.split("Final_Output:")[1]

        # 2. Remove unnecessary leading/trailing whitespace or line breaks from the string.
        cleaned_string = list_string.strip()

        # 3. Safely convert the cleaned string into an actual Python list.
        final_list = ast.literal_eval(cleaned_string)

        return final_list

    except (IndexError, SyntaxError, ValueError):
        # In cases such as when "Final_Output:" is missing or the list format is invalid,
        # safely return an empty list for all exceptional situations.
        return []

def scope_question_to_dict(text_data: str) -> Dict[int, str]:

    parsed_dict = {}

    # --- Modified regular expression ---
    # (.*?): Non-greedily finds the text after 'Answer'.
    # (?=\s*\d+\.|\Z): Sets the boundary to search only up to just before the 'next number' or the 'end of the string'.
    pattern = re.compile(r"(\d+)\..*?Answer:?\s*(.*?)(?=\s*\d+\.|\Z)", re.DOTALL | re.IGNORECASE)

    # Use findall to find all (key, value) pairs across the entire text at once.
    matches = pattern.findall(text_data)

    for key, value in matches:
        # Add the found result to the dictionary.
        parsed_dict[int(key)] = value.strip()
        
    return parsed_dict

def analyze_medical_data(text):
    """
    Final function that clarifies the label rules and preserves all of the original return format.
    """
    lines = text.strip().split('\n')

    date_regex = re.compile(r'\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]')
    # Clarify the rule so that the label 'starts with an alphabet letter'
    label_regex = re.compile(r"^\s*\d+:?\s+([A-Za-z][A-Za-z0-9\s]*)\s*$")
    number_regex = re.compile(r"^\s*\d+:?\s+([0-9\.]+)\s*$")

    date_count = 0
    item_counts = {}
    current_item = None

    for line in lines:
        if date_regex.search(line):
            date_count += 1
            continue
        
        line_stripped = line.strip()

        label_match = label_regex.match(line_stripped)
        if label_match:
            current_item = label_match.group(1).strip()
            if current_item not in item_counts:
                item_counts[current_item] = 0
            continue
        
        number_match = number_regex.match(line_stripped)
        if number_match and current_item:
            item_counts[current_item] += 1

    # Finally, find the items whose count matches the date count. (original code restored)
    matching_items = [item for item, count in item_counts.items() if count == date_count]

    # Return all three values as originally intended. (original code restored)
    return date_count, item_counts, matching_items

def tag_text_range_corrected(data_dict, start_keyword, end_keyword):
    result_dict = {}
    is_in_range = False

    try:
        sorted_keys = sorted(data_dict.keys(), key=lambda x: int(x.split('_')[1]))
    except (ValueError, IndexError):
        sorted_keys = sorted(data_dict.keys())

    for key in sorted_keys:
        text = data_dict[key]

        if start_keyword in text:
            is_in_range = True

        if end_keyword in text:
            is_in_range = False
        # --- End of logic modification ---

        result_dict[key] = {
            'text': text,
            'in_range': is_in_range
        }
            
    return result_dict

def extract_data_from_text(text):

    flowsheet_pattern = r"Flowsheet Data as of\s*(\[.*?\]\s*\d{1,2}:\d{2}\s*(?:AM|PM))"
    
    # 2. String pattern after 'Since'
    # e.g.: 12 AM
    since_pattern = r"Since\s+(.*)"

    # Use a regular expression to search in the text
    flowsheet_match = re.search(flowsheet_pattern, text)
    since_match = re.search(since_pattern, text)

    # If there is a search result, take the first group (the content inside the parentheses); otherwise assign None
    flowsheet_data = flowsheet_match.group(1).strip() if flowsheet_match else None
    since_data = since_match.group(1).strip() if since_match else None
    if since_data:
        if '12' in since_data:
            since_data = "12 AM"
        elif '*0*' in since_data:
            since_data = "0 AM"
        else:
            for time_c in range(25,100):
                if str(time_c) in since_data:
                    since_data = None
                    break
            

    return [flowsheet_data, since_data]


def find_entity_line_number_lab(subtext, entity):

  # Split the text into individual lines.
  lines = subtext.strip().split('\n')

  # Iterate over each line and check.
  for line in lines:
    # Split into the line number part and the content part on ':'.
    parts = line.split(':', 1)
    if len(parts) == 2:
      line_number = parts[0].strip()
      content = parts[1].strip()

      # Check whether the content part matches the entity being searched for.
      if content == entity:
        # If it matches, convert the line number to an integer and return it.
        return int(line_number)

  # If the entity is not found after checking everything, return None.
  return None


def process_cluster_output(text: str) -> list:
    pattern = re.compile(r"^Line number\s*(\d+)\.\s*(.*?):\s*\[(.*?)]\s*-\s*\[(.*?)]$", re.MULTILINE)

    matches = pattern.findall(text)

    processed_list = []
    for match in matches:
        line_num = match[0]
        entity = match[1].strip()
        value = match[2].strip()
        
        # Make the linking-entity list into a list by splitting on commas (,).
        linking_entities_str = match[3].strip()
        linking_entities = [e.strip() for e in linking_entities_str.split(',')]

        processed_list.append([entity, line_num, value, linking_entities])

    return processed_list

def parse_gemini_output(input_str: str) -> List[list]:
    # Strip leading/trailing whitespace from the input string and return an empty list if it does not start with "Line number".
    try:
        if not input_str.strip().startswith("Line number"):
            return []
    except:
        return []

    # List to store the results
    results = []

    # Regular expression pattern definition
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

def main():
    parser = argparse.ArgumentParser(description="Process a clinical note based on YAML config.")
    parser.add_argument("config_path", help="Path to the YAML configuration file")
    args = parser.parse_args()
    config = load_config(args.config_path)

    note_type = "physician"                    # fixed per script
    testval = os.getenv("TEST_VAL", "val")     # {"val", "test"}
    test_val_dir = 'valid' if testval == 'val' else testval
    df = pd.read_csv(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/{note_type}_{testval}.csv')
    vertex_model = init_vertex_model()
    azure_client = init_azure_client()
    model_name_file = 'gemini'
    deployment = os.getenv("AZURE_DEPLOYMENT")
    NOTE_IDX = os.getenv("NOTE_IDX")  # e.g. NOTE_IDX=6 to run one note; unset = run all
    for idx, note in enumerate(df["TEXT"]):
        if NOTE_IDX is not None and idx != int(NOTE_IDX):
            continue
        hadmid = df['HADM_ID'][idx]
        print('hadm id:', hadmid)
        extracted_info=extract_data_from_text(note)
        flow_sheet_date = datetime.strptime(re.sub(r'[\[\*\]]', '', extracted_info[0]).strip(), "%Y-%m-%d %I:%M %p").strftime("%Y-%m-%d %H:%M:%S")
        flow_sheet_date_only=flow_sheet_date.split(' ')[0]
        from_time_flow=extracted_info[1]
        if from_time_flow:
            from_time_flow = from_time_flow.replace('**','').replace('[','').replace(']','').replace('a.m.', 'am')
            from_time_flow_24h=datetime.strptime(from_time_flow, '%I %p').strftime('%H:%M:%S')
        else:
            from_time_flow_24h=None
        print("date:  ")
        print(flow_sheet_date,from_time_flow)
        adm_time=df['ADMITTIME'][idx]
        chart_time=df['CHARTDATE'][idx]
        adm_time_only_date = re.search(r'\d{4}-\d{2}-\d{2}', adm_time).group(0)
        chart_time_only_date = re.search(r'\d{4}-\d{2}-\d{2}', chart_time).group(0)
        
        with open("../tool_calling/current_hadm_id.txt", "w", encoding="utf-8") as f:
            f.write(f"{note_type}, {testval}, {hadmid}")
        
        target_hadm_id = hadmid.astype(int)
        record_data_df = pd.read_csv(f'{EHR_REASONCON_DIR}/patient_data/{target_hadm_id}/patient_records.csv')
        record_data_df['TOP_VALUES'] = record_data_df['TOP_VALUES'].str[:-1] + ", 'NaN']"
        record_str_list = [f"{label}: {values}" for label, values in zip(record_data_df['LABEL'], record_data_df['TOP_VALUES'])]
        record_str_list = "\n".join(record_str_list)

        ## df for data that the actual patient does not have in the db (added)
        remain_data_df = pd.read_csv(f'{EHR_REASONCON_DIR}/patient_data/{target_hadm_id}/catalog_items.csv')
        remain_data_df['TOP_VALUES'] = remain_data_df['TOP_VALUES'].str[:-1] + ", 'NaN']"

        numbered_note = add_line_numbers(note)
        list_subtext_temp=extract_medical_text_indices_fully_revised(numbered_note)
        list_subtext_temp=[tup for tup in list_subtext_temp if None not in tup]
        list_subtext_temp= [f'{start}~{end}' for start, end in list_subtext_temp]
        if len(list_subtext_temp) ==1:
            subtext_prompt = open_file(config["prompts"]["extract_subtext"]).replace("<<<Clinical_Note>>>", numbered_note).replace(", and combine <<<SCOPE2>>> into another subtext","").replace(" one subtext and <<<SCOPE2>>> another subtext","").replace("<<<SCOPE1>>>",list_subtext_temp[0])
        else:
            subtext_prompt = open_file(config["prompts"]["extract_subtext"]).replace("<<<Clinical_Note>>>", numbered_note).replace("<<<SCOPE1>>>",list_subtext_temp[0]).replace("<<<SCOPE2>>>",list_subtext_temp[1])
        gemini_response = gemini_inference(vertex_model, "gemini-2.5-flash", subtext_prompt)
        final_output_str = correct_segmentation_final(gemini_response, list_subtext_temp)
        
        ner_time_result = []
        sub= extract_subtext(numbered_note, final_output_str) 
        with open(f'./output/subtext_divided_{note_type}_{testval}_{idx}_{model_name_file}_2.pkl', 'wb') as f:
            pickle.dump(sub, f)

        sub_modified = tag_text_range_corrected(sub, start_keyword='Flowsheet', end_keyword='Labs / Radiology')

        for key, data in sub_modified.items():
            subtext_ = data['text']
            is_in_range  = data['in_range']
            temp_dict_line_num = create_line_mapping(subtext_, subtext_)
            if 'Total In:' in subtext_:
                continue
            else:
                find_lab_prompt=open_file(config['prompts']['find_lab_in_subtext']).replace("<<<CLINICAL_NOTE>>>",subtext_)
                yes_or_no = gemini_inference(vertex_model, "gemini-2.5-flash", find_lab_prompt)
                if 'yes' in yes_or_no.lower() and 'Other labs:' not in subtext_ and 'Labs / Radiology' not in subtext_:  ## need to check the format written in note 3 and fix it
                    d_count, i_counts, count_same_lab=analyze_medical_data(subtext_)
                    if len(count_same_lab) > 0:
                        focused_entity= str(count_same_lab)
                        lab_ner_prompt=open_file(config['prompts']['lab_ner']).replace("<<<CLINICAL_NOTE>>>",subtext_).replace("<<<FOCUSED_ENTITY>>>",focused_entity)
                        lab_ner_result=gemini_inference(vertex_model, "gemini-2.5-flash", lab_ner_prompt)
                        lab_final_result_=extract_list_from_text(lab_ner_result)
                        print(len(lab_final_result_))
                        print("lab_ner:  ", lab_final_result_) ##[time, lab name, result]
                        for ilab_f in lab_final_result_:
                            line_numbers=find_entity_line_number_lab(subtext_,ilab_f[1])
                            line_numbers_str = str(line_numbers)
                            text_info = f'{ilab_f[0]} {ilab_f[1]} {ilab_f[2]}' 
                            result_entity = f"{{{ilab_f[1]}:{ilab_f[2]}}}"
                            list_a = [line_numbers_str, text_info, text_info, result_entity, [ilab_f[0]]]
                            ner_time_result.append(list_a)
                    else:
                        print(d_count, i_counts, count_same_lab)
                        continue
                else:   
                    if 'flowsheet data as of' in subtext_.lower():
                        prompt_cluster_record_prompt=open_file('./prompt/cluster/record_cluster_phy.txt').replace("<<<PREDEFINED_VALUES>>>",record_str_list).replace("<<<CLINICAL_NOTE>>>",subtext_)
                    else:
                        prompt_cluster_record_prompt=open_file('./prompt/cluster/record_cluster.txt').replace("<<<PREDEFINED_VALUES>>>",record_str_list).replace("<<<CLINICAL_NOTE>>>",subtext_)
                    gemini_record_cluster = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_record_prompt)

                    final_ner_output1=process_cluster_output(gemini_record_cluster)  ##result of entity extraction based on the content stored in the patient's ehr [[entity, linenumber, value, [litem1, item2,... ]],[entity, linenumber, value, [litem1, item2,... ]]]
                    seen = set()
                    final_ner_output1 = [item[:2] + [str(item[2]) + ".0", item[3]] if (key := (item[0], item[1], item[2], tuple(item[3]))) in seen else (seen.add(key) or item) for item in final_ner_output1]
                    str_final_ner_output1 = '\n'.join([str(item) for item in final_ner_output1]) #convert final_ner_output1 to str
                    ## case that is not in the patient's ehr but only in mimic
                    unique_groups =remain_data_df['GROUP'].unique()  ##first extract all groups
                    alphabet = string.ascii_uppercase

                    # Use enumerate to get the index and value together and combine into the desired format
                    formatted_list = [f"({alphabet[i]}) {group}" for i, group in enumerate(unique_groups)] #A) Group1 B) Group2... make multiple-choice options from the groups
                    print("Let's start ehr table")
                    # Join all items in the list with a space (" ") into a single string
                    result_string = " ".join(formatted_list)
                    prompt_cluster_remain_prompt=open_file('./prompt/cluster/cluster_group_quiz.txt').replace("<<<SELECTED_CATEGORY>>>",result_string).replace("<<<CLINICAL_NOTE>>>",subtext_)
                    gemini_remain_cluster = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_remain_prompt)
                    flattened_remain_subgroup = []


                    if gemini_remain_cluster.strip() != '[]':
                        result_list_remain = [re.sub(r'^\s*\([A-Z]\)\s*', '', item).strip() for item in gemini_remain_cluster.strip('[]').split(',')] #make the gemini_remain_cluster result into a list
                        final_df_subgroup = remain_data_df[remain_data_df['GROUP'].isin(result_list_remain)] ##find GROUP from the extracted ones -> make a new df (consisting only of the extracted groups)
                        unique_groups_subgroup=final_df_subgroup['SUBGROUP'].unique() ## extract subgroups uniquely based on the selected GROUP

                        # Use enumerate to get the index and value together and combine into the desired format
                        formatted_list_ = [f"({i+1}) {group}" for i, group in enumerate(unique_groups_subgroup)] # 1)subgroup 2)subgroup make options
                        result_string_subgroup = " ".join(formatted_list_)
                        prompt_cluster_remain_prompt_subgroup=open_file('./prompt/cluster/cluster_group_quiz.txt').replace("<<<SELECTED_CATEGORY>>>",result_string_subgroup).replace("<<<CLINICAL_NOTE>>>",subtext_)
                        gemini_remain_cluster_subgroup = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_remain_prompt_subgroup)
                        if gemini_remain_cluster_subgroup.strip() != '[]':
                            try:
                                subgroup_list_output=[s.split(')', 1)[1].strip() for s in gemini_remain_cluster_subgroup.strip('[]').split(', (')]  ##make the selected subgroups into a list
                            except:
                                subgroup_list_output=[]
                            subgroup_df = remain_data_df[remain_data_df['SUBGROUP'].isin(subgroup_list_output)] ##make a new df using only the selected subgroups
                            subgroup_df=subgroup_df.reset_index(drop=True)
                            subgroup_label_list = subgroup_df['LABEL'].astype(str).tolist() ## extract the LABEL values from the df newly built by subgroup
                            chunk_size = 200
                            separator = ", "  # separator to put between items (e.g. ", " or " | ")
                            chunked_strings = [separator.join(subgroup_label_list[i:i + chunk_size]) for i in range(0, len(subgroup_label_list), chunk_size)] ##split into small lists of 200 each
                            print(len(chunked_strings))
                            final_remain_subgroup_total=[]
                            for c_id, chunk_ in enumerate(chunked_strings):
                                print("NOW ON2:  ",c_id)
                                final_remain_subgroup_prompt=open_file('./prompt/cluster/cluster_quiz1.txt').replace("<<<CLINICAL NOTE>>>",subtext_).replace("<<<CHECKED LIST>>>",chunk_).replace("<<<LAST_EXTRACTED>>>",str_final_ner_output1)
                                final_remain_subgroup_output = gemini_inference(vertex_model, "gemini-2.5-flash", final_remain_subgroup_prompt)
                                final_remain_subgroup_list=parse_gemini_output(final_remain_subgroup_output)
                                final_remain_subgroup_list = [item for item in final_remain_subgroup_list if item[-1] != [''] and item[-1] != []]
                                final_remain_subgroup_total.append(final_remain_subgroup_list)
                                final_remain_subgroup_total = [item for item in final_remain_subgroup_total if item]
                            flattened_remain_subgroup = [item for sublist in final_remain_subgroup_total for item in sublist]
                        else:
                            print("case where the 2nd (subgroup selection) is an empty list")
                            ### needs to be written here
                    else:
                        print("when the 1st is an empty list, skip everything and merge")
                    combined_list = flattened_remain_subgroup + final_ner_output1  
                    final_ner_output_ = sorted(combined_list, key=lambda x: int(x[1])) ## linenumber sorting 
                    final_ner_output = []
                    for item in final_ner_output_: 
                        if item[:-1] not in final_ner_output:
                            final_ner_output.append(item[:-1])  ##remove the part corresponding to item  [[entity, linenumber,– value, [litem1, item2,... ]] -> [[[entity, linenumber, value],]

                        ## it would be good to remove Keywords here!!
                    keywords_ner = ['bpm','hour','units/hour','TSICU','female','male','down','monday','emergency','ed','transfers','transer','time','assessed','unchanged','resident','history','her','his','services','physically','chief complaint','chief','complaint','ICU medications','family','pm','am','since','since 12 am','hours','24 hours','hemodynamic monitoring','fluid balance','other icu medications','other medications','last dose of antibiotics','data','flowsheet','examination','physical','physical examination','flowsheet data','other','icu','dose','last','last dose','chief complaint','chief','complaint','24 hour events','events','24 hour','~','recommendations','events','blood','chemistry','urine','superficial','AM','yesterday','~','episode','measures','single','Supportive measures','family history','review of systems','admission','review of','review','medical','medications','changes','change','labs','radiology','other labs','assessment','plan','assessment and plan']
                    lower_keywords_set = {k.lower() for k in keywords_ner}

                    # 2. Filtering (this part is the actual one-line code)
                    final_ner_output = [item for item in final_ner_output if item[0].lower() not in lower_keywords_set]

                    # Remove those whose first element is a number
                    final_ner_output = [sublist for sublist in final_ner_output if not sublist[0].replace('.', '', 1).isdigit()]
                                            
                    ## is this okay here? #ner_time_result
                    parsed_data = parse_medical_record(subtext_)
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

                    result_parsing=combine_data(parsed_data, formmated_ner_ouput)
                    if len(result_parsing)==0:
                        print(result_parsing) # <- commented out for now
                    for _r in result_parsing:
                        scope_prompt=open_file('./prompt/scope_phy.txt').replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<CLINICAL NOTE>>>",subtext_).replace("<<<Admission>>>",adm_time_only_date).replace("<<<Chartdate>>>",chart_time_only_date).replace("<<<CLINICAL_NOTE_CONTEXT>>>",_r[2])
                        scope_output = gemini_inference(vertex_model, "gemini-2.5-flash", scope_prompt)
                        lines = scope_output.split("\n")
                        try:
                            result = {
                                num.strip(): val.strip() 
                                for num, val in (line.split(":", 1) for line in lines if ":" in line)
                            }
                        except:
                            print("scope_output: ", scope_output)
                        if any(val.lower() == "yes" for val in result.values()): ## caught by scope
                            print("Removed")
                            print(_r[3]) 
                            continue
                        else:
                            if is_in_range == True:
                                print(_r[3])
                                if 'Tmax' in _r[3] and from_time_flow_24h:
                                    start_exp = flow_sheet_date_only+" "+from_time_flow_24h 
                                    end_exp = flow_sheet_date
                                    time_expression = f'{start_exp}-{end_exp}'
                                    print("Tamx time expression:  ",time_expression)
                                    print()
                                    _r.append([time_expression])
                                    ner_time_result.append(_r)
                                elif 'Tcurrent' in _r[3] and from_time_flow_24h:
                                    start_exp = flow_sheet_date_only+" "+from_time_flow_24h 
                                    end_exp = flow_sheet_date
                                    time_expression = f'Last record from {start_exp} between {end_exp}.'
                                    print("TCurrent time expression:  ",time_expression)
                                    print()
                                    _r.append([time_expression])
                                    ner_time_result.append(_r)
                                elif 'admission' in _r[3].lower():
                                    get_range = lambda d_str: ((datetime.strptime(d_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'), (datetime.strptime(d_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
                                    start, end = get_range(adm_time.split(' ')[0])
                                    time_expression = f'from {start} to {end}'
                                    print("admission time expression:  ",time_expression)
                                    print()
                                    _r.append([time_expression])
                                    ner_time_result.append(_r)
                                else:
                                    time_expression = f"{(datetime.strptime(flow_sheet_date, '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')}-{flow_sheet_date}"
                                    print("Time exp:   ", time_expression)
                                    print()
                                    _r.append([time_expression])
                                    ner_time_result.append(_r)
                            else:
                                ent_ = re.findall(r"\{(.+?):", _r[3])
                                print(ent_)
                                time1_prompt = open_file("./prompt/physician/time_1.txt").replace("<<<ENTITY>>>",ent_[0]).replace("<<<CLINICAL_NOTE>>>",subtext_)
                                time1_response=vertex_model.models.generate_content(model="gemini-2.5-flash", contents=time1_prompt, config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=256)),)
                                print(time1_response.text)
                                if 'yes' in time1_response.text:
                                    time2_prompt = open_file("./prompt/physician/time_2.txt").replace("<<<ENTITY>>>",ent_[0]).replace("<<<CLINICAL_NOTE>>>",subtext_).replace("<<<ADM_DATE>>>",adm_time_only_date).replace("<<<CHART_DATE>>>",chart_time_only_date)
                                    time2_response=vertex_model.models.generate_content(model="gemini-2.5-flash", contents=time2_prompt, config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=256)),)
                                    time_expression=time2_response.text
                                    print("time_expression", time_expression)
                                    _r.append([time_expression])
                                    ner_time_result.append(_r)
                                else: 
                                    get_range = lambda d_str: ((datetime.strptime(d_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'), (datetime.strptime(d_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
                                    start, end = get_range(chart_time_only_date)
                                    time_expression = f'from {start} to {end}'  ##chartdate -1 ~ chartdate +1
                                    print("admission time expression:  ",time_expression)
                                    _r.append([time_expression])
                                    ner_time_result.append(_r)

        with open(f'./output/clusterner_time_result_{note_type}_{testval}_{idx}_v2_{model_name_file}_2.pkl', 'wb') as f:
            pickle.dump(ner_time_result, f)

                

if __name__ == "__main__":
    main()
 