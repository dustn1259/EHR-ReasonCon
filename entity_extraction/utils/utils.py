import yaml
import re

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def add_line_numbers(text):
    lines = text.split('\n')
    return '\n'.join(f"{i}:\t{line}" for i, line in enumerate(lines))

def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_subtext(text, information):
    """
    Extract portions of text based on line number ranges from a formatted information string.
    
    Args:
        text (str): The input text to process
        information (str): A multi-line string with ranges in the format "- 0~5"
    
    Returns:
        dict: Dictionary containing extracted text portions with keys as "subject_{index}"
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Process the information string to extract ranges
    # Remove any leading/trailing whitespaces and split by new lines
    info_lines = [line.strip() for line in information.strip().split('\n')]
    
    # Extract only the ranges (remove the bullet points)
    ranges = []
    for line in info_lines:
        # Skip empty lines
        if not line:
            continue
        # Extract the range part (after the hyphen and whitespace)
        if line.startswith('-'):
            range_str = line[1:].strip()
            if '~' in range_str:  # Verify it's a valid range
                ranges.append(range_str)
    
    # Dictionary to store the results
    subjects = {}
    
    # Process each range
    for i, range_str in enumerate(ranges):
        # Parse the range
        start, end = map(int, range_str.split('~'))
        
        # Extract the lines for this range
        # Make sure we don't go out of bounds
        start = max(0, start)
        end = min(len(lines) - 1, end)
        
        # Join the lines for this range
        subject_text = '\n'.join(lines[start:end+1])
        
        # Store in the dictionary
        subjects[f"subject_{i}"] = subject_text
    
    return subjects

def parse_medical_record(text):
    """
    Parse a medical record text with line numbers into a dictionary.
    
    Args:
        text (str): The medical record text with line numbers
        
    Returns:
        dict: A dictionary with line numbers as keys and line content as values
    """
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Initialize an empty dictionary
    result = {}
    
    # Process each line
    for line in lines:
        # Check if the line starts with a number followed by a colon
        if line and line[0].isdigit() and ':' in line:
            # Split by the first colon
            parts = line.split(':', 1)
            line_num = parts[0].strip()
            line_content = parts[1].strip() if len(parts) > 1 else ""
            
            # Add to dictionary
            result[line_num] = line_content
        else:
            # Handle continuation lines or empty lines
            # If there's no line number, append to the previous line
            if result:
                last_key = list(result.keys())[-1]
                result[last_key] += "\n" + line
            else:
                # If this is the first line and doesn't have a number
                result['0'] = line
    
    return result

def list_to_string(input_list):
    return ','.join(f"'{item}'" for item in input_list)

def ner_string_format(ner_string):
    """
    Process NER string formats and return a clean, consistent string format
    that preserves all key-value pairs including duplicates.
    
    Args:
        ner_string (str): The input NER string in various formats
        
    Returns:
        str: Formatted string with all key-value pairs in "{key: value}" format
    """
    # Check if input is empty or not a string
    if not ner_string or not isinstance(ner_string, str):
        return ""
    
    # First, remove any markdown code block markers (```)
    if "```" in ner_string:
        # Remove all instances of ```
        lines = ner_string.split('\n')
        filtered_lines = []
        for line in lines:
            if line.strip() == "```" or line.strip().startswith("```"):
                # Skip lines that are just code block markers
                continue
            filtered_lines.append(line)
        ner_string = '\n'.join(filtered_lines)
    
    # Clean the input string
    cleaned_string = ner_string.strip()
    key_value_pairs = []
    
    # Check for json format: [{"key": "value"}]
    if "[" in cleaned_string and "]" in cleaned_string and "{" in cleaned_string and "}" in cleaned_string:
        # Extract data inside brackets
        start_idx = cleaned_string.find("[")
        end_idx = cleaned_string.rfind("]")
        if start_idx != -1 and end_idx != -1:
            json_content = cleaned_string[start_idx+1:end_idx].strip()
            
            # Parse each item in the format: { "key: value" }
            items = []
            
            # Handle nested braces more carefully
            start = 0
            brace_count = 0
            for i, char in enumerate(json_content):
                if char == '{':
                    brace_count += 1
                    if brace_count == 1:
                        start = i
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        items.append(json_content[start:i+1])
            
            for item in items:
                item = item.strip('{}').strip()
                if ":" in item:
                    # Handle quoted keys like "Lung_(breath sounds): Absent"
                    if item.startswith('"') and '"' in item[1:]:
                        quote_end = item.find('"', 1)
                        if quote_end != -1 and quote_end < len(item) - 1:
                            if item[quote_end+1:].strip().startswith(":"):
                                key = item[1:quote_end].strip()
                                value = item[item.find(":", quote_end)+1:].strip()
                                # Remove quotes from value if present
                                value = value.strip('"')
                                key_value_pairs.append((key, value))
                    else:
                        parts = item.split(":", 1)
                        key = parts[0].strip().strip('"')
                        value = parts[1].strip().strip('"')
                        key_value_pairs.append((key, value))
    
    # Check for the brace format: {key: value} {key: value}
    elif "{" in cleaned_string and "}" in cleaned_string:
        # Identify all pairs of braces
        i = 0
        segments = []
        start = -1
        
        while i < len(cleaned_string):
            if cleaned_string[i] == '{':
                start = i
            elif cleaned_string[i] == '}' and start != -1:
                segments.append(cleaned_string[start+1:i])
                start = -1
            i += 1
        
        for segment in segments:
            if ":" in segment:
                parts = segment.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                key_value_pairs.append((key, value))
    
    # Format all key-value pairs into a consistent string
    result_parts = []
    for key, value in key_value_pairs:
        result_parts.append(f"{{{key}: {value}}}")
    
    result_string = " ".join(result_parts)
    return result_string


def format_ner_data(data_list):
    result = ""
    
    for item in data_list:
        line_num = item['line_num']
        ner_output = item['ner_output']
        
        result += f"{line_num}:\n"
        result += f"output: {ner_output}\n\n"
    
    # Remove trailing newlines
    return result.rstrip()

import re

def combine_data(parsed_data, final_ner_output,context_lines=1):
    """
    Combines text from parsed_data with entities from final_ner_output,
    including context lines before and after each matched line.
    
    Args:
        parsed_data (dict): Dictionary with line numbers as keys and text content as values.
        final_ner_output (str): String containing line numbers and entity outputs.
        context_lines (int): Number of lines to include before and after the matched line.
        
    Returns:
        list: List of lists, each containing [line_number, line_text, context_text, entity].
    """
    result = []
    
    # Remove markdown code blocks if present
    final_ner_output = re.sub(r'```.*?```', lambda m: m.group(0).replace('```', ''), final_ner_output, flags=re.DOTALL)
    final_ner_output = final_ner_output.replace('```', '').strip()
    
    # Convert parsed_data keys to integers for easier manipulation
    parsed_data_int_keys = {int(k): v for k, v in parsed_data.items()}
    
    # Find min and max line numbers in parsed_data
    min_line = min(parsed_data_int_keys.keys()) if parsed_data_int_keys else 0
    max_line = max(parsed_data_int_keys.keys()) if parsed_data_int_keys else 0
    
    # Extract line numbers and entities using regex
    pattern = r'(\d+):\s*output:\s*((?:\{[^{}]*\}\s*)*)'
    matches = re.finditer(pattern, final_ner_output)
    
    for match in matches:
        line_number = int(match.group(1))
        output_text = match.group(2)
        
        # Extract all entities (patterns like {key: value})
        entities = re.findall(r'(\{[^{}]+?\})', output_text)
        
        # Skip empty entities like {: } or {:}
        valid_entities = []
        for entity in entities:
            if re.match(r'\{\s*:\s*\}', entity) or re.match(r'\{\s*:\s*\}', entity):
                continue
            valid_entities.append(entity)
        
        # Skip if no valid entities
        if not valid_entities:
            continue
        
        # Get the text for the current line
        line_text = parsed_data_int_keys.get(line_number, "")
        
        # Get context lines (current line plus surrounding lines)
        context_content = []
        
        # Calculate the range of lines to include
        start_line = max(line_number - context_lines, min_line)
        end_line = min(line_number + context_lines, max_line)
        
        # Collect text from all context lines
        for ctx_line in range(start_line, end_line + 1):
            if ctx_line in parsed_data_int_keys:
                context_content.append(parsed_data_int_keys[ctx_line])
        
        # Join the context lines with spaces
        combined_context = " ".join(context_content)
        
        for entity in valid_entities:
            result.append([str(line_number), line_text, combined_context, entity])
    
    return result

def find_subjects_by_line_number(sub_time, line_num):
    """
    Function that finds the most appropriate subject among the subjects containing the given line_num.

    Args:
        sub_time (dict): Dictionary containing text for each subject
        line_num (str): The line number to find (as a string)

    Returns:
        str: The selected subject key (e.g., 'subject_0')
    """
    import re

    # Step 1: Find the subjects that contain line_num
    target_subjects = []
    line_pattern = f"^{line_num}:\t"  # Match the "number:\t" pattern at the start of the line

    for subject_key, text in sub_time.items():
        # Check line by line
        lines = text.split('\n')
        for line in lines:
            if re.match(line_pattern, line):
                target_subjects.append(subject_key)
                break

    # If no subject contains the given line_num, return None
    if not target_subjects:
        return None

    # Step 2: Find the start and end line numbers of each subject
    subject_scores = []

    for subject_key in target_subjects:
        text = sub_time[subject_key]
        lines = text.split('\n')

        # Extract the line numbers
        line_numbers = []
        for line in lines:
            # Find the "number:\t" pattern at the start of the line
            match = re.match(r'^(\d+):\t', line)
            if match:
                line_numbers.append(int(match.group(1)))

        if line_numbers:
            start_line = min(line_numbers)
            end_line = max(line_numbers)
            target_line = int(line_num)

            # Step 3: Calculate the score (line_num - start) + (end - line_num)
            score = (target_line - start_line) + (end_line - target_line)
            subject_scores.append((subject_key, score))

    # Step 4: Select the subject with the highest score (if tied, select the first)
    if subject_scores:
        # Sort in descending order by score; if scores are equal, keep the original order
        subject_scores.sort(key=lambda x: x[1], reverse=True)
        return subject_scores[0][0]
    
    return None

    
def parse_entity_string_structured(input_text):
    """
    Parses the input string and converts it into a list of [entity, line_number, associated_values].

    [Modified] Removes single quotes from both ends of associated_values if present.

    Input format: "[{line_number;;entity1;;'values1'}, {line_number;;entity2;;values2}, ...]"
    Output format: [ ['entity1', 'line_number', 'values1'], ['entity2', 'line_number', 'values2'], ... ]
    """

    output_list = []

    # 1. Remove the '[' and ']' at both ends
    if not input_text.startswith('[') or not input_text.endswith(']'):
        return output_list

    content = input_text[1:-1]

    # 2. Split each item by ','
    if not content:
        return output_list

    elements = content.split('}, {') # Split by comma and braces

    for element in elements:
        try:
            # 3. Remove the '{' and '}' and whitespace from each item
            cleaned_element = element.strip().strip('{}')

            # 4. Split into 3 parts by ';;'
            parts = cleaned_element.split(';;')

            if len(parts) == 3:
                line_number = parts[0]
                entity = parts[1]

                # 5. [Modified] Process assoc_values: remove single quotes from both ends
                assoc_values = parts[2].strip("'")

                # 6. Change the order and add to the list
                output_list.append([entity, line_number, assoc_values])
            else:
                print(f"Warning: skipping an item with an invalid format: {element}")

        except Exception as e:
            print(f"Error occurred: while processing {element} {e}")
            
    return output_list