import re
def extract_medical_text_indices_fully_revised(medical_text: str) -> list[tuple[int, int]]:
    """
    Extracts two line number ranges from medical text based on specific patterns.
    (Final version reflecting all requirements)
    """
    lines = medical_text.strip().split('\n')

    # Initialize variables to store the results
    flowsheet_line = None
    total_in_line = None
    lab_data_start_line = None
    lab_data_end_line = None

    # Define regex patterns
    date_time_pattern = re.compile(r'\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]\s+\d{1,2}:\d{2}\s+(AM|PM)')
    wbc_pattern = re.compile(r'\bWBC\b') # Pattern to find the word 'WBC' itself

    # First loop: find the positions of the main lines.
    for i, line in enumerate(lines):
        if 'Flowsheet Data as of' in line and '[' in line and ']' in line:
            flowsheet_line = i

        # 👇 This is the modified part!
        # Store the line number just before the 'Total In:' line.
        if 'Total In:' in line:
            total_in_line = i - 1
            
        if 'Other labs:' in line and lab_data_end_line is None:
            lab_data_end_line = i - 1

    # Logic to search for the Lab Data start point
    for i in range(len(lines)):
        is_first_date_in_sequence = date_time_pattern.search(lines[i]) and \
                                    (i == 0 or not date_time_pattern.search(lines[i-1]))

        if is_first_date_in_sequence:
            temp_index = i
            while (temp_index + 1 < len(lines) and 
                   date_time_pattern.search(lines[temp_index + 1])):
                temp_index += 1
            
            if (temp_index + 1 < len(lines) and 
                wbc_pattern.search(lines[temp_index + 1])):
                lab_data_start_line = i
                break

    # Combine the final results
    range1 = (flowsheet_line, total_in_line)
    range2 = (lab_data_start_line, lab_data_end_line)

    return [range1, range2]

def correct_segmentation_final(initial_str, mandatory_list):
    """
    Takes an initial segmentation list given as a single string and
    a mandatory segmentation list (list), resolves all boundary and
    overlap issues, and returns the final result as a string.

    Args:
        initial_str (str): The initial segmentation string in the format '- 0~3\n- 4~7...'.
        mandatory_list (list): The mandatory segmentation list in the format ['10~13', ...].

    Returns:
        str: The final re-segmented result string.
    """
    initial_list = [line for line in initial_str.strip().split('\n') if line.strip()]

    def _parse_range(range_str):
        print(f"DEBUG: incoming value = '{range_str}'") # Added this line to check!
        clean_str = range_str.strip().lstrip('-').strip()
        start, end = map(int, clean_str.split('~'))
        return start, end

    all_numbers = []
    for r_str in initial_list + mandatory_list:
        start, end = _parse_range(r_str)
        all_numbers.extend([start, end])

    if not all_numbers:
        return ""

    max_boundary = max(all_numbers)
    # 0: empty, positive: initial range with a unique ID, negative: mandatory range
    line_map = [0] * (max_boundary + 2)

    # 🐞 Bug fix point 🐞
    # 1. Draw the sketch (assign a "unique ID" to each initial range)
    range_id = 1
    for r_str in initial_list:
        start, end = _parse_range(r_str)
        for i in range(start, end + 1):
            line_map[i] = range_id
        range_id += 1 # Increase the ID for the next range

    # 2. Overpaint (overwrite mandatory ranges with the special ID -1)
    MANDATORY_ID = -1
    for r_str in mandatory_list:
        start, end = _parse_range(r_str)
        for i in range(start, end + 1):
            line_map[i] = MANDATORY_ID

    # 3. Check the results (now accurately reconstructed thanks to the unique IDs)
    final_ranges = []
    i = 0
    while i < len(line_map):
        if line_map[i] == 0:
            i += 1
            continue
        
        segment_type = line_map[i]
        segment_start = i
        j = i
        while j < len(line_map) and line_map[j] == segment_type:
            j += 1
        segment_end = j - 1
        final_ranges.append(f"{segment_start}~{segment_end}")
        i = j
        
    formatted_lines = [f"- {r}" for r in final_ranges]
    return '\n'.join(formatted_lines)