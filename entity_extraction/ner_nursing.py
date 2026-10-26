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
        # Use a regular expression to find all numeric values (including decimals).

        # From the result of split('/'), keep only the items that have a value (non-empty) in the values list.
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
        # Intersection check: they intersect if current's start is <= last's end
        if current[0] <= last[1]:
            # Merge: extend the end point to the larger value
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

def standardize_text(text: str) -> str:
    """Standardizes text for comparison (lowercase, strip leading/trailing whitespace)."""
    return text.lower().strip()


def extract_numbered_lines(text: str) -> List[Tuple[int, str]]:
    """Extracts the text into a list of (line number, content)."""
    lines = []
    for line in text.strip().split('\n'):
        match = re.match(r'(\d+):\s*(.*)', line)
        if match:
            lines.append((int(match.group(1)), match.group(2).strip()))
    return lines

def standardize_text_new(text: str) -> str:
    """Standardizes text for comparison (lowercase, strip leading/trailing whitespace + normalize internal whitespace)."""
    s = text.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def _canonicalize_keep_map(s: str):
    """
    Returns a 'normalized string' for comparison (keeping only alphanumerics) and
    a list mapping each normalized character index -> original text index.
    """
    canon_chars = []
    idx_map = []
    for i, ch in enumerate(s):
        if ch.isalnum():
            canon_chars.append(ch)
            idx_map.append(i)
    return ''.join(canon_chars), idx_map

def create_line_mapping(subtext_check: str, output_check_number: str) -> Dict[int, List[int]]:
    # 1) Parse
    sub_lines = extract_numbered_lines(subtext_check)
    out_lines = extract_numbered_lines(output_check_number)

    # 2) Expand (standardize) subtext + record cumulative length at line boundaries
    std_sub_texts = [standardize_text_new(t) for _, t in sub_lines]
    sep = " "
    pieces = []
    line_start_pos = []
    pos = 0
    for idx, txt in enumerate(std_sub_texts):
        line_start_pos.append(pos)
        pieces.append(txt)
        pos += len(txt)
        if idx != len(std_sub_texts) - 1:
            pieces.append(sep)
            pos += len(sep)
    flat_sub = "".join(pieces)

    # 3) Record the [start, end) range of each subline
    spans = []
    for i, (lnum, _) in enumerate(sub_lines):
        start = line_start_pos[i]
        end = start + len(std_sub_texts[i])
        spans.append((start, end, lnum))

    def charpos_to_subline_nums(start_pos: int, end_pos: int) -> List[int]:
        touched = []
        for s, e, lnum in spans:
            if not (end_pos <= s or e <= start_pos):
                touched.append(lnum)
        return sorted(set(touched))

    # 4) Normalization for ignoring whitespace/punctuation + index map
    canon_flat, flat_map = _canonicalize_keep_map(flat_sub)

    # 5) Search each Output line in order (allow overlap, compare ignoring whitespace/punctuation)
    mapping: Dict[int, List[int]] = {}
    search_from_canon = 0  # search start position based on the normalized string
    for out_num, out_text in out_lines:
        needle = standardize_text_new(out_text)
        if not needle:
            continue
        canon_needle, _ = _canonicalize_keep_map(needle)

        if not canon_needle:
            continue

        found_at_canon = canon_flat.find(canon_needle, search_from_canon)

        if found_at_canon == -1 and search_from_canon > 0:
            # Defensively try one position back as well (in case it slightly overlaps the previous match)
            found_at_canon = canon_flat.find(canon_needle, max(0, search_from_canon - 1))

        if found_at_canon != -1:
            # Convert the normalized index to the original text (flat_sub) index
            start_orig = flat_map[found_at_canon]
            end_orig = flat_map[found_at_canon + len(canon_needle) - 1] + 1
            mapping[out_num] = charpos_to_subline_nums(start_orig, end_orig)

            # Allow overlap: keep the next search start point at this match's "start"
            search_from_canon = found_at_canon
        # If not found, just skip

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

        # 1) A '_number' suffix uses the previous location
        if re.search(r'_\d+$', entity):
            entity_locations.append({entity: last_real_location or -1})
            continue

        regex = make_pattern(entity)
        found = False

        # ---------- 2) Search the output region ----------
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

            if subtext_lines:           # if there is a mapping, use it as-is
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
                    if sub_ln < start_sub:          # start looking from the same line
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
    Finds the corresponding location (line number) in the text based on a list of NER entities.

    - If the same 'base entity name' appears consecutively, copies the location of the first entity.
    - The 'base entity name' is the name excluding parentheses and their contents. (e.g., 'Wgt (current)' -> 'Wgt')
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
        # Remove parentheses and their contents to extract the 'base entity name'. (e.g., 'Wgt (current)' -> 'Wgt')
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

        # --- 1. Search for the entity in the output region ---
        for out_line_num in sorted(output_map):
            if out_line_num < search_line_num:
                continue

            line = output_map[out_line_num]
            start_pos = search_char_idx if out_line_num == search_line_num else 0

            match = regex.search(line[start_pos:])
            if not match:
                continue

            # --- 2. Map the output location to a subtext location ---
            subtext_lines = mapping.get(out_line_num)
            final_sub_line = -1

            if subtext_lines:
                # If there are multiple mapped subtext lines, prefer the line that contains the entity name
                final_sub_line = next(
                    (ln for ln in subtext_lines if regex.search(subtext_map.get(ln, ''))),
                    subtext_lines[0] # if not found, use the first mapped line
                )
            else:
                # If there is no mapping information, search directly across the entire subtext
                start_search_sub_line = last_real_location if last_real_location is not None else -1
                for sub_ln in sorted(subtext_map):
                    if sub_ln < start_search_sub_line:
                        continue
                    if regex.search(subtext_map[sub_ln]):
                        final_sub_line = sub_ln
                        break
                # If not found in the subtext, use the output line number instead (fallback policy)
                if final_sub_line == -1:
                    final_sub_line = out_line_num

            # --- 3. Save the result and update the cursor for the next search ---
            entity_locations.append({entity: final_sub_line})
            last_real_location = final_sub_line
            search_line_num = out_line_num
            search_char_idx = match.end() + start_pos
            found = True
            break # search for the current entity succeeded, move on to the next entity

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
    # 2. Build the final list while assigning values sequentially
    output_list = []
    # A counter that tracks which value index to take for each key ('HR', 'BP')
    value_counters = {key: 0 for key in parsed_data.keys()}

    for item in original_list:
        key = item[0]
        line_num = item[1]

        # Get the values to assign
        all_values = parsed_data.get(key)

        value_to_assign = [] # default value
        if all_values:
            # Get the counter corresponding to the current key
            current_index = value_counters.get(key, 0)

            # Check whether there are values left to assign
            if current_index < len(all_values):
                value_to_assign = all_values[current_index]
                # Increment this key's counter by 1 so the next value is taken next time
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
    Uses a regular expression to flexibly handle whitespace and line breaks.
    """
    try:
        # 1. Find the pattern '[', optional whitespace, '[', any characters (.), ']', optional whitespace, ']'.
        # The re.DOTALL option makes the '.' character also match newline characters (\n).
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
        # 1. Split the text on "Final_Output:" and take the latter part ([1]).
        list_string = text_input.split("Final_Output:")[1]

        # 2. Remove unnecessary leading/trailing whitespace or line breaks from the string.
        cleaned_string = list_string.strip()

        # 3. Safely convert the cleaned string into an actual Python list.
        final_list = ast.literal_eval(cleaned_string)

        return final_list

    except (IndexError, SyntaxError, ValueError):
        # Cases such as "Final_Output:" being absent or the list format being invalid, etc.
        # In all exceptional cases, safely return an empty list.
        return []

def scope_question_to_dict(text_data: str) -> Dict[int, str]:

    parsed_dict = {}

    # --- Modified regular expression ---
    # (.*?): non-greedily finds the text after 'Answer'.
    # (?=\s*\d+\.|\Z): sets a boundary so it only matches up to just before the 'next number' or the 'end of string'.
    pattern = re.compile(r"(\d+)\..*?Answer:?\s*(.*?)(?=\s*\d+\.|\Z)", re.DOTALL | re.IGNORECASE)

    # Use findall to find all (key, value) pairs across the entire text at once.
    matches = pattern.findall(text_data)

    for key, value in matches:
        # Add the found result to the dictionary.
        parsed_dict[int(key)] = value.strip()
        
    return parsed_dict

def analyze_medical_data(text):
    """
    The final function that clarifies the label rules and preserves all of the original return formats.
    """
    lines = text.strip().split('\n')

    date_regex = re.compile(r'\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]')
    # Clarify the rule so that labels must 'start with an alphabet letter'
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


def parse_gemini_output(input_str: str) -> List[list]:
    # Strip leading/trailing whitespace from the input string and return an empty list if it does not start with "Line number".
    if not input_str.strip().startswith("Line number"):
        return []

    # List to store the results
    results = []

    # Define the regular expression pattern
    # The meaning of each group is as follows:
    # 1. (\d+): line number (one or more digits)
    # 2. (.*?): entity (all characters from after '. ' up to before ' - [', minimal match)
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

def extract_data_from_text(text):

    flowsheet_pattern = r"Flowsheet Data as of\s*(\[.*?\]\s*\d{1,2}:\d{2}\s*(?:AM|PM))"
    
    # 2. The string pattern after 'Since'
    # e.g.: 12 AM
    since_pattern = r"Since\s+(.*)"

    # Search the text using regular expressions
    flowsheet_match = re.search(flowsheet_pattern, text)
    since_match = re.search(since_pattern, text)

    # If there is a match, take the first group (the content inside the parentheses); otherwise assign None
    flowsheet_data = flowsheet_match.group(1).strip() if flowsheet_match else None
    since_data = since_match.group(1).strip() if since_match else None

    return [flowsheet_data, since_data]


def find_entity_line_number_lab(subtext, entity):

  # Split the text into individual lines.
  lines = subtext.strip().split('\n')

  # Iterate over each line and check.
  for line in lines:
    # Split into the line number part and the content part based on ':'.
    parts = line.split(':', 1)
    if len(parts) == 2:
      line_number = parts[0].strip()
      content = parts[1].strip()

      # Check whether the content part matches the entity being searched for.
      if content == entity:
        # If it matches, convert the line number to an integer and return it.
        return int(line_number)

  # If the entity is not found after checking the whole text, return None.
  return None


def process_cluster_output(text: str) -> list:
    pattern = re.compile(r"^Line number\s*(\d+)\.\s*(.*?):\s*\[(.*?)]\s*-\s*\[(.*?)]$", re.MULTILINE)

    matches = pattern.findall(text)

    processed_list = []
    for match in matches:
        line_num = match[0]
        entity = match[1].strip()
        value = match[2].strip()
        
        # Split the linking entity list by comma (,) to make it a list.
        linking_entities_str = match[3].strip()
        linking_entities = [e.strip() for e in linking_entities_str.split(',')]

        processed_list.append([entity, line_num, value, linking_entities])

    return processed_list


def adjust_ner_lines(final_ner_output, temp_dict_line_num, subtext_):
    # Convert subtext_ into a per-line text dictionary
    lines = {}
    for ln in str(subtext_).splitlines():
        if ":" in ln:
            i, t = ln.split(":", 1)
            lines[int(i.strip())] = t.strip().lower()

    for item in final_ner_output:
        text, key = item[0], int(item[1])
        candidates = temp_dict_line_num.get(key, [])
        if len(candidates) == 1:  # if there is one element, use it as-is
            item[1] = str(candidates[0])
        elif len(candidates) > 1:  # if there are several, find the position in subtext_
            t = text.lower().split()[0]  # based on the first word
            found = next((i for i in candidates if t in lines.get(i, "")), candidates[0])
            item[1] = str(found)
    return final_ner_output


def main():
    parser = argparse.ArgumentParser(description="Process a clinical note based on YAML config.")
    parser.add_argument("config_path", help="Path to the YAML configuration file")
    args = parser.parse_args()
    config = load_config(args.config_path)

    note_type = "nursing"                      # fixed per script
    testval = os.getenv("TEST_VAL", "val")     # {"val", "test"}
    test_val_dir = 'valid' if testval == 'val' else testval
    df = pd.read_csv(f'{EHR_REASONCON_DIR}/{note_type}/{test_val_dir}/{note_type}_{testval}.csv')
    vertex_model = init_vertex_model()
    model_name_file = 'gemini'
    NOTE_IDX = os.getenv("NOTE_IDX")  # e.g. NOTE_IDX=6 to run one note; unset = run all
    for idx, note in enumerate(df["TEXT"]):
        if NOTE_IDX is not None and idx != int(NOTE_IDX):
            continue
        extracted_info=extract_data_from_text(note)
        adm_time=df['ADMITTIME'][idx]
        chart_time=df['CHARTDATE'][idx]
        adm_time_only_date = re.search(r'\d{4}-\d{2}-\d{2}', adm_time).group(0)
        chart_time_only_date = re.search(r'\d{4}-\d{2}-\d{2}', chart_time).group(0)
        hadmid = df['HADM_ID'][idx]
        with open("../tool_calling/current_hadm_id.txt", "w", encoding="utf-8") as f:
            f.write(f"{note_type}, {testval}, {hadmid}")

        target_hadm_id = hadmid.astype(int)
        record_data_df = pd.read_csv(f'{EHR_REASONCON_DIR}/patient_data/{target_hadm_id}/patient_records.csv')
        record_data_df['TOP_VALUES'] = record_data_df['TOP_VALUES'].str[:-1] + ", 'NaN']"
        record_str_list = [f"{label}: {values}" for label, values in zip(record_data_df['LABEL'], record_data_df['TOP_VALUES'])]
        record_str_list = "\n".join(record_str_list)

        ## df for data that is not in the db for the actual patient (added)
        remain_data_df = pd.read_csv(f'{EHR_REASONCON_DIR}/patient_data/{target_hadm_id}/catalog_items.csv')
        remain_data_df['TOP_VALUES'] = remain_data_df['TOP_VALUES'].str[:-1] + ", 'NaN']"


        numbered_note = add_line_numbers(note)
        subtext_prompt = open_file(config["prompts"]["extract_subtext"]).replace("<<<Clinical_Note>>>", numbered_note)
        gemini_response = gemini_inference(vertex_model, "gemini-2.5-flash", subtext_prompt)

        ner_time_result = []
        sub= extract_subtext(numbered_note, gemini_response) 
        with open(f'./output/subtext_divided_{note_type}_{testval}_{idx}_{model_name_file}_2.pkl', 'wb') as f:
            pickle.dump(sub, f)

        for i in sub:
            subtext_ = sub[i]
            if 'Valuables / Signature' in subtext_:
                continue
            subtext_check = subtext_
            subtext_with_line = subtext_  # version with line numbers preserved
            subtext_no_line  = re.sub(r"^\d+:\s*", "", subtext_with_line, flags=re.MULTILINE)

            reformat_prompt = open_file('./prompt/ner_v2/reformat_prompt.txt').replace("<<<DISCHARGE_SUMMARY>>>", subtext_no_line)
            output = gemini_inference(vertex_model, "gemini-2.5-flash", reformat_prompt)
            output_check_number = add_line_numbers(output)  # (same, just for checking)
            temp_dict_line_num = create_line_mapping(subtext_check, output_check_number)
            subtext_ner = re.sub(r'^\d+:\s*', '', output, flags=re.MULTILINE)  # (same, for NER)
            
            prompt_cluster_record_prompt=open_file('./prompt/cluster/record_cluster.txt').replace("<<<PREDEFINED_VALUES>>>",record_str_list).replace("<<<CLINICAL_NOTE>>>",output_check_number)
            gemini_record_cluster = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_record_prompt)

            final_ner_output1=process_cluster_output(gemini_record_cluster)  ##result of entity extraction based on content stored in the patient's ehr [[entity, linenumber, value, [litem1, item2,... ]],[entity, linenumber, value, [litem1, item2,... ]]]
            str_final_ner_output1 = '\n'.join([str(item) for item in final_ner_output1]) #convert final_ner_output1 to str
            ## case where it is not in the patient's ehr but only in mimic
            unique_groups =remain_data_df['GROUP'].unique()  ##first extract all groups
            alphabet = string.ascii_uppercase

            # use enumerate to get the index and value together and combine them into the desired format
            formatted_list = [f"({alphabet[i]}) {group}" for i, group in enumerate(unique_groups)] #A) Group1 B) Group2... make multiple-choice options from the groups
            print("Let's start ehr table")
            # join all items of the list with a space (" ") to make a single string
            result_string = " ".join(formatted_list)
            prompt_cluster_remain_prompt=open_file('./prompt/cluster/cluster_group_quiz.txt').replace("<<<SELECTED_CATEGORY>>>",result_string).replace("<<<CLINICAL_NOTE>>>",output_check_number)
            gemini_remain_cluster = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_remain_prompt)
            flattened_remain_subgroup = []


            if gemini_remain_cluster.strip() != '[]':
                result_list_remain = [re.sub(r'^\s*\([A-Z]\)\s*', '', item).strip() for item in gemini_remain_cluster.strip('[]').split(',')] #turn the gemini_remain_cluster result into a list
                final_df_subgroup = remain_data_df[remain_data_df['GROUP'].isin(result_list_remain)] ##find GROUP from the extracted ones -> make a new df (consisting only of the extracted groups)
                unique_groups_subgroup=final_df_subgroup['SUBGROUP'].unique() ## uniquely extract subgroups based on the selected GROUPs

                # use enumerate to get the index and value together and combine them into the desired format
                formatted_list_ = [f"({i+1}) {group}" for i, group in enumerate(unique_groups_subgroup)] # make options like 1)subgroup 2)subgroup
                result_string_subgroup = " ".join(formatted_list_)
                prompt_cluster_remain_prompt_subgroup=open_file('./prompt/cluster/cluster_group_quiz.txt').replace("<<<SELECTED_CATEGORY>>>",result_string_subgroup).replace("<<<CLINICAL_NOTE>>>",output_check_number)
                gemini_remain_cluster_subgroup = gemini_inference(vertex_model, "gemini-2.5-pro", prompt_cluster_remain_prompt_subgroup)

                if gemini_remain_cluster_subgroup.strip() != '[]':
                    try:
                        subgroup_list_output=[s.split(')', 1)[1].strip() for s in gemini_remain_cluster_subgroup.strip('[]').split(', (')]  ##turn the selected subgroups into a list
                    except:
                        pass
                    subgroup_df = remain_data_df[remain_data_df['SUBGROUP'].isin(subgroup_list_output)] ##make a new df using only the selected subgroups
                    subgroup_df=subgroup_df.reset_index(drop=True)
                    subgroup_label_list = subgroup_df['LABEL'].astype(str).tolist() ## extract the LABEL values from the newly created subgroup df
                    chunk_size = 200
                    separator = ", "  # delimiter placed between items (e.g., ", " or " | ")
                    chunked_strings = [separator.join(subgroup_label_list[i:i + chunk_size]) for i in range(0, len(subgroup_label_list), chunk_size)] ##split into small lists of 200 each
                    print(len(chunked_strings))
                    final_remain_subgroup_total=[]
                    for c_id, chunk_ in enumerate(chunked_strings):
                        print("NOW ON2:  ",c_id)
                        final_remain_subgroup_prompt=open_file('./prompt/cluster/cluster_quiz1.txt').replace("<<<CLINICAL NOTE>>>",output_check_number).replace("<<<CHECKED LIST>>>",chunk_).replace("<<<LAST_EXTRACTED>>>",str_final_ner_output1)
                        final_remain_subgroup_output = gemini_inference(vertex_model, "gemini-2.5-flash", final_remain_subgroup_prompt)
                        final_remain_subgroup_list=parse_gemini_output(final_remain_subgroup_output)
                        final_remain_subgroup_list = [item for item in final_remain_subgroup_list if item[-1] != [''] and item[-1] != []]
                        final_remain_subgroup_total.append(final_remain_subgroup_list)
                        final_remain_subgroup_total = [item for item in final_remain_subgroup_total if item]
                    flattened_remain_subgroup = [item for sublist in final_remain_subgroup_total for item in sublist]
                else:
                    print("case where the 2nd (subgroup extraction) is an empty list")
                    ### needs to be written here
            else:
                print("when the 1st is an empty list, skip everything and merge")
            combined_list = flattened_remain_subgroup + final_ner_output1
            final_ner_output_ = sorted(combined_list, key=lambda x: int(x[1])) ## linenumber sorting
            final_ner_output = []
            for item in final_ner_output_:
                if item[:-1] not in final_ner_output:
                    final_ner_output.append(item[:-1])  ##remove the corresponding item part  [[entity, linenumber,– value, [litem1, item2,... ]] -> [[[entity, linenumber, value],]
            final_ner_output_11=final_ner_output
            #output_check_number
            final_ner_output=adjust_ner_lines(final_ner_output, temp_dict_line_num, subtext_)


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

            formmated_ner_ouput = f"```\n{formmated_ner_ouput}```"
            result_parsing=combine_data(parsed_data, formmated_ner_ouput)
            if len(result_parsing)==0:
                print(result_parsing) # <- commented out for now
            for _r in result_parsing:
                scope_prompt=open_file('./prompt/scope_nur.txt').replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<CLINICAL NOTE>>>",subtext_check).replace("<<<Admission>>>",adm_time_only_date).replace("<<<Chartdate>>>",chart_time_only_date).replace("<<<CLINICAL_NOTE_CONTEXT>>>",_r[2])
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
                    continue
                else:
                    print("usable entity:   ",_r[3])
                    print()
                    time_selection_prompt = open_file('./prompt/nurse/time_prompt.txt').replace("<<<NURSING_NOTE>>>",subtext_).replace("<<<LINE>>>",_r[0]).replace("<<ENTITY>>",re.search(r'\{([^:]+):', _r[3]).group(1))
                    time_selection_output=vertex_model.models.generate_content(model="gemini-2.5-flash", contents=time_selection_prompt, config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=256)),).text
                    print(time_selection_output)
                    if 'a' == time_selection_output.lower():
                        time_output_final = f"{(datetime.strptime(adm_time_only_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')}~{(datetime.strptime(adm_time_only_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')}"
                    elif 'b' == time_selection_output.lower():
                        standard_time_prompt = open_file('./prompt/nurse/b_time_prompt.txt').replace("<<<NURSING_NOTE>>>",subtext_check).replace("<<<LINE>>>",_r[0]).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1))
                        time_output_final=vertex_model.models.generate_content(model="gemini-2.5-flash", contents=standard_time_prompt, config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=256)),).text
                    else:
                        time_output_final = f"{(datetime.strptime(chart_time_only_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')}~{(datetime.strptime(chart_time_only_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')}"
                    print(time_output_final)
                    ner_time_result.append(list(_r) + [[time_output_final]])
                    continue ## entities that are used
                         
        with open(f'./output/clusterner_time_result_{note_type}_{testval}_{idx}_v2_{model_name_file}_2.pkl', 'wb') as f:
            pickle.dump(ner_time_result, f)

                
 
if __name__ == "__main__":
    main()
 