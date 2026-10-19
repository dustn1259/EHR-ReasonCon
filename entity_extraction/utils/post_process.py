def extract_text_before_colon(input_string):
    responses = {}
    lines = input_string.strip().split('\n')

    for line in lines:
        # Split into "number: answer" format
        if ': ' in line:
            parts = line.split(': ')
        elif '. ' in line:
            parts = line.split('. ')
        if len(parts) == 2:
            try:
                number = int(parts[0])
                answer = parts[1].strip()
                responses[number] = answer
            except:
                print(f"❗️ ValueError occurred! Starting the debugger.")
    if responses[1] == 'Yes':
        return 'No'
    else:
        has_yes = any(responses.get(i, '').lower() == 'yes' for i in range(2, 18))
        result = 'Yes' if has_yes else 'No'
        return result

    


def filter_line_ranges(line_number_info, reference_ranges):
    """
    Filters the reference ranges that fall within the given start-end line number range.

    Args:
        line_number_info (str): A string in the format 'Start line number: X\nEnd line number: Y'
        reference_ranges (str or list):
            - If a string: format '- 0~6\n- 8~11\n...'
            - If a list: format ['0~7', '8~11', ...]

    Returns:
        list: The filtered list of ranges
    """
    # Extract the start and end numbers from line_number_info
    lines = line_number_info.replace('ber:\n','ber:').strip().split('\n')
    try:
        start_line = int(lines[0].split(':')[-1].strip())
        end_line = int(lines[1].split(':')[-1].strip())
    except:
        pass

    # Convert reference_ranges to a list if it is a string
    if isinstance(reference_ranges, str):
        # Split each line and remove '- '
        range_list = []
        for line in reference_ranges.strip().split('\n'):
            line = line.strip()
            if line.startswith('- '):
                range_list.append(line[2:])  # Remove '- '
            elif line:  # If not an empty line
                range_list.append(line)
        reference_ranges = range_list
    
    filtered_ranges = []
    
    for range_str in reference_ranges:
        if not range_str.strip():  # Skip empty strings
            continue
        # Extract the start and end numbers from the range string (supports both ~ and -)
        try:
            if '~' in range_str:
                parts = range_str.split('~')
                parts = [p.strip() for p in parts if p.strip()]  # Remove empty strings and whitespace
                if len(parts) == 2:
                    start_ref, end_ref = map(int, parts)
                else:
                    continue
            elif '-' in range_str:
                parts = range_str.split('-')
                parts = [p.strip() for p in parts if p.strip()]  # Remove empty strings and whitespace
                if len(parts) == 2:
                    start_ref, end_ref = map(int, parts)
                else:
                    continue
            else:
                # Skip if there is no separator
                continue
        except ValueError:
            # Skip if it cannot be converted to a number
            continue

        # Check whether the ranges overlap
        # Condition: when the reference range overlaps with the given range
        if not (end_ref < start_line or start_ref > end_line):
            filtered_ranges.append(range_str.strip())
    
    return filtered_ranges


def extend_ranges(ranges):
    """
    Generates extended segments by connecting each segment with the preceding and following segments.

    Args:
        ranges (list): A list of ranges in the format ['7~11', '12~18', '20~23', ...]

    Returns:
        list: The list of extended ranges
    """
    if len(ranges) < 2:
        return ranges.copy()
    
    extended_ranges = []
    
    for i in range(len(ranges)):
        # Extract the start and end of the current segment
        current_range = ranges[i]
        if '~' in current_range:
            current_start, current_end = map(int, current_range.split('~'))
        elif '-' in current_range:
            current_start, current_end = map(int, current_range.split('-'))
        else:
            continue
        
        # Calculate the start and end of the range to extend
        extend_start = current_start
        extend_end = current_end
        
        # If not the first: connect with the previous segment
        if i > 0:
            prev_range = ranges[i-1]
            if '~' in prev_range:
                prev_start, _ = map(int, prev_range.split('~'))
            elif '-' in prev_range:
                prev_start, _ = map(int, prev_range.split('-'))
            else:
                prev_start = current_start
            extend_start = prev_start
        
        # If not the last: connect with the next segment
        if i < len(ranges) - 1:
            next_range = ranges[i+1]
            if '~' in next_range:
                _, next_end = map(int, next_range.split('~'))
            elif '-' in next_range:
                _, next_end = map(int, next_range.split('-'))
            else:
                next_end = current_end
            extend_end = next_end
        
        # Add the extended range (using the same separator as the original)
        if '~' in current_range:
            extended_ranges.append(f"{extend_start}~{extend_end}")
        else:
            extended_ranges.append(f"{extend_start}-{extend_end}")
    
    return extended_ranges

def merge_with_unfiltered(line_number_info, gemini_response_text, extended, filtered):
    """
    Finds the unfiltered ranges, merges them with extended, and sorts them.

    Args:
        line_number_info (str): A string in the format 'Start line number: X\nEnd line number: Y'
        gemini_response_text (str): A string in the format '- 0~5\n- 6~11\n...'
        extended (list): The list of extended ranges
        filtered (list): The list of filtered ranges

    Returns:
        list: A list of the unfiltered ranges + extended ranges sorted by start number
    """
    # Extract all ranges from gemini_response_text
    all_ranges = []
    if isinstance(gemini_response_text, str):
        for line in gemini_response_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('- '):
                all_ranges.append(line[2:])  # Remove '- '
            elif line:  # If not an empty line
                all_ranges.append(line)

    # Find the unfiltered ranges
    unfiltered_ranges = []
    for range_str in all_ranges:
        if range_str.strip() and range_str.strip() not in filtered:
            unfiltered_ranges.append(range_str.strip())
    
    # Merge extended and unfiltered_ranges
    combined_ranges = extended.copy() + unfiltered_ranges

    # Sort by start number
    def get_start_number(range_str):
        try:
            if '~' in range_str:
                return int(range_str.split('~')[0])
            elif '-' in range_str:
                return int(range_str.split('-')[0])
            else:
                return float('inf')  # If parsing is not possible, move to the end
        except:
            return float('inf')
    
    combined_ranges.sort(key=get_start_number)
    
    return combined_ranges


def parse_string_to_list(string_list_str: str) -> list:
    content = string_list_str.strip('[]')
    items = content.split(';;')
    cleaned_list = [item.strip() for item in items]

    return cleaned_list