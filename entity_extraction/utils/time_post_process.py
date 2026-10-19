import re
import sys
sys.path.insert(0, '.')  # change to the actual path
from models import init_vertex_model, init_azure_client, get_azure_response, get_azure_response2
from google.genai import types
from datetime import datetime, timedelta

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def parse_medical_output(input_text):
    """Parse the medical analysis output and return a structured result"""

    def extract(pattern, text):
        """Extract text with a regular expression"""
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    # Extract the answer and reason for each question
    q1_reason = extract(r'Question1\..*?\nReason:\s*(.+?)(?=\nAnswer:)', input_text)
    q1_answer = extract(r'Question1\..*?Answer:\s*(.+?)(?=\nQuestion[2-6]|$)', input_text)
    
    q2_reason = extract(r'Question2\..*?\nReason:\s*(.+?)(?=\nAnswer:)', input_text)
    q2_answer = extract(r'Question2\..*?\nAnswer:\s*(.+?)(?=\nQuestion[3-6]|\ndate_start |$)', input_text)


    # Question3 data
    date_start  = extract(r'date_start -\s*(.+?)(?=\n|$)', input_text)
    date_end  = extract(r'date_end -\s*(.+?)(?=\n|$)', input_text)
    ndays = extract(r'n-days-\s*(.+?)(?=\n|$)', input_text)

    q4_reason = extract(r'Question4\..*?\nReason:\s*(.+?)(?=\nAnswer:)', input_text)
    q4_answer = extract(r'Question4\..*?\nAnswer:\s*(.+?)(?=\nQuestion[5-6]|$)', input_text)


    q5_reason = extract(r'Question5\..*?\nReason:\s*(.+?)(?=\nAnswer:)', input_text)
    q5_answer = extract(r'Question5\..*?\nAnswer:\s*(.+?)(?=\nQuestion6|$)', input_text)
    
    # Add Question6
    q6_reason = extract(r'Question6\..*?\nReason:\s*(.+?)(?=\nAnswer:)', input_text)
    q6_answer = extract(r'Question6\..*?\nAnswer:\s*(.+?)(?=\n|$)', input_text)

    # Extract the specific answer
    def get_answer_value(text):
        if not text:
            return None
        if 'Yes' in text:
            return 'Yes'
        elif 'No' in text:
            return 'No'
        elif 'inferred' in text:
            return 'inferred'
        elif 'single' in text:
            return 'single'
        elif 'Category 1' in text:
            return 'Category 1'
        elif 'Category 2' in text:
            return 'Category 2'
        elif 'Category 3' in text:
            return 'Category 3'
        # Add Question6 answer types
        elif 'adm' in text.lower():
            return 'adm'
        elif 'dis' in text.lower():
            return 'dis'
        elif 'tod' in text.lower():
            return 'tod'
        elif 'yesterday' in text.lower():
            return 'yesterday'
        elif 'pod#' in text.lower():
            # Extract pod#3 form
            pod_match = re.search(r'pod#(\d+)', text.lower())
            return f'pod#{pod_match.group(1)}' if pod_match else 'pod#'
        elif 'hod#' in text.lower():
            # Extract hod#5 form
            hod_match = re.search(r'hod#(\d+)', text.lower())
            return f'hod#{hod_match.group(1)}' if hod_match else 'hod#'
        elif 'n/a' in text.lower():
            return 'n/a'
        return text.strip()

    # Track the analysis flow
    q1_val = get_answer_value(q1_answer)
    
    result = {
        'answers': {
            'q1': q1_val,
            'q2': get_answer_value(q2_answer),
            'q3': None,  # Question3 handled specially
            'q4': get_answer_value(q4_answer),
            'q5': get_answer_value(q5_answer),
            'q6': get_answer_value(q6_answer)
        },
        'dates': {
            'start': date_start  if date_start  and date_start  != '-' else None,
            'end': date_end  if date_end  and date_end  != '-' else None,
            'duration': ndays if ndays and ndays != '-' else None
        },
        'reasons': {
            'q1': q1_reason,
            'q2': q2_reason,
            'q4': q4_reason,
            'q5': q5_reason,
            'q6': q6_reason
        }
    }

    # Question3 special handling - in the Both case, if date info exists, activate q3 too
    if result['answers']['q2'] == 'inferred':
        if any([date_start , date_end , ndays]):
            result['answers']['q3'] = 'completed'

    # Add the analysis end point (add Q5, Q6 branching logic)
    if q1_val == 'No':
        result['stopped_at'] = 'q1'
    elif result['answers']['q2'] == 'inferred':
        result['stopped_at'] = 'q3'
    elif result['answers']['q4'] == 'Category 1':
        if result['answers']['q5'] == 'Yes':
            result['stopped_at'] = 'q5'
        elif result['answers']['q5'] == 'No':
            result['stopped_at'] = 'q6'
        else:
            result['stopped_at'] = 'q5'
    elif result['answers']['q4'] in ['Category 2', 'Category 3']:
        result['stopped_at'] = 'q4'
    else:
        result['stopped_at'] = 'unknown'

    return result

def q3_data_parser(start, end, duration):
    if duration is None:
        text = f"Start on {start} and end on {end}."
    else:
        text = f"Start on {start}, end on {end}, with a duration of {duration}."
    return text

def filtering(output_dict,entity,line_num,time_summary,hos,note,adm_time,chart_time,vertex_model=None):
    if output_dict['answers']['q1'] == 'No':
        return "no time expression"
    else:
        if output_dict['answers']['q2'] == 'inferred':
            #This vancomycin-related event occurred within the timeframe spanning from 2011-11-11 to 2011-11-17
            if output_dict['dates']['duration'] is None:
                text = f'This {entity}-related event occurred within the timeframe spanning from {{{{output_dict["dates"]["start"]}}}} to {{{{output_dict["dates"]["end"]}}}}'
            else:
                text = f"This {entity}-related event occurred within the timeframe spanning from {output_dict['dates']['start']} to {output_dict['dates']['end']} (duration: {output_dict['dates']['duration']})"
            with open('../prompt/time_prompt/inferred.txt', 'r', encoding='utf-8') as file:
                content = file.read()
            content=content.replace("<<<BEFORE_REVESION>>>",text).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary)
            time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                        contents=content, 
                                        config=types.GenerateContentConfig(
                                                thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                ),)

            return time_output.text
        else:
            if output_dict['answers']['q4'] == 'Category 1':
                if output_dict['answers']['q5'] == 'No':
                    if output_dict['answers']['q6'] == 'adm':
                        return f'From one day before {adm_time} to the day after {adm_time}'
                    elif output_dict['answers']['q6'] == 'dis':
                        return f'From one day before {chart_time} to the day after {chart_time}'
                    elif output_dict['answers']['q6'] == 'yesterday':
                        try:
                            new_date = (datetime.strptime(chart_time, '%Y-%m-%d') + timedelta(days=-1)).strftime('%Y-%m-%d')
                        except: 
                            new_date = (datetime.strptime(chart_time, '%Y-%m-%d %H:%M:%S') + timedelta(days=-1)).strftime('%Y-%m-%d')
                        return f'From one day before {new_date} to the day after {new_date}'
                    elif 'hod#' in output_dict['answers']['q6']:
                        pod_match = re.search(r'pod#(\d+)', output_dict['answers']['q6'].lower())
                        try:
                            days=pod_match.group(1)
                        except:
                            days = 0
                        try:
                            new_date = (datetime.strptime(adm_time, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')
                        except:
                            new_date = (datetime.strptime(adm_time, '%Y-%m-%d %H:%M:%S') + timedelta(days=-days)).strftime('%Y-%m-%d')
                        return f'From one day before {new_date} to the day after {new_date}'
                    elif 'pod#' in output_dict['answers']['q6']:
                        with open('../prompt/time_prompt/hod.txt', 'r', encoding='utf-8') as file:
                            content = file.read()
                        content=content.replace("<<<POD_OUTPUT>>>",output_dict['answers']['q6']).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary)
                        time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                            contents=content, 
                                                            config=types.GenerateContentConfig(
                                                                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                    ),)
            

                        if "we don't know" in time_output.text:
                            time_output= 'no time expression'
                        else:
                            time_output= time_output.text
                        return time_output
                    elif output_dict['answers']['q6'] == 'tod':
                        with open('../prompt/time_prompt/tod.txt', 'r', encoding='utf-8') as file:
                            content = file.read() 
                        content=content.replace("<<<CLINICAL NOTE>>>",note).replace("<<<ENTITY>>>",entity).replace("<<<LINE_NUMBER>>>",line_num).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary).replace("<<<TIME_ANALYSIS>>>",hos)
                        time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                            contents=content, 
                                                            config=types.GenerateContentConfig(
                                                                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                    ),)

                        return time_output.text

                    elif output_dict['answers']['q6'] == 'n/a':
                        with open('../prompt/time_prompt/time_yyyy_mm_dd_make.txt', 'r', encoding='utf-8') as file:
                            content = file.read()
                        content=content.replace("<<<start_or_end>>>","start").replace("<<<ENTITY>>>",entity).replace("<<<LINE_NUMBER>>>",line_num).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary).replace("<<<CLINICAL_NOTE>>>",note).replace("<<<hos>>>",hos)
                        time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                            contents=content, 
                                                            config=types.GenerateContentConfig(
                                                                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                    ),)

                        return time_output.text
                else:
                    with open('../prompt/time_prompt/time_yyyy_mm_dd.txt', 'r', encoding='utf-8') as file:
                        content = file.read()
                    content=content.replace("<<<start_or_end>>>","start").replace("<<<ENTITY>>>",entity).replace("<<<LINE_NUMBER>>>",line_num).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary).replace("<<<CLINICAL_NOTE>>>",note).replace("<<<hos>>>",hos)
                    time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                        contents=content, 
                                                        config=types.GenerateContentConfig(
                                                                thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                ),)
                    return time_output.text
            elif output_dict['answers']['q4'] == 'Category 2':
                with open('../prompt/time_prompt/time_yyyy_mm_dd_make.txt', 'r', encoding='utf-8') as file:
                        content = file.read()
                content = content=content.replace("<<<start_or_end>>>","end").replace("<<<ENTITY>>>",entity).replace("<<<LINE_NUMBER>>>",line_num).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary).replace("<<<CLINICAL_NOTE>>>",note).replace("<<<hos>>>",hos)
                time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                        contents=content, 
                                                        config=types.GenerateContentConfig(
                                                                thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                ),)

                return time_output.text
                
            else:
                with open('../prompt/time_prompt/time_n_days.txt', 'r', encoding='utf-8') as file:
                        content = file.read()
                content = content=content.replace("<<<start_or_end>>>","end").replace("<<<ENTITY>>>",entity).replace("<<<LINE_NUMBER>>>",line_num).replace("<<<TEMPORTAL_EXPRESSION>>>",time_summary).replace("<<<CLINICAL_NOTE>>>",note).replace("<<<hos>>>",hos)
                time_output = vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                                        contents=content, 
                                                        config=types.GenerateContentConfig(
                                                                thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                ),)
                return time_output.text
