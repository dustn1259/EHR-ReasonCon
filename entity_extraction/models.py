import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types
from utils.utils import *
from datetime import datetime, timedelta

load_dotenv()

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _creds:  # only export when set; a missing .env must not break `import models`
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds

def init_vertex_model():
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
    return client

def init_azure_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    )


def get_azure_response(client, prompt, model):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant analyzing clinical notes."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=model,
    )
    return response.choices[0].message.content


def get_azure_response2(client, prompt, model):
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        model=model,
    )
    return response.choices[0].message.content



def get_day_before_and_after(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    day_before = date - timedelta(days=1)
    day_before = day_before.strftime("%Y-%m-%d")
    day_after = date + timedelta(days=1)
    day_after = day_after.strftime("%Y-%m-%d")

    return f"{day_before} to {day_after}"


def gemini_inference(vertex_model, model_name, input):
    if model_name == 'gemini-2.5-flash':
        return vertex_model.models.generate_content(model="gemini-2.5-flash", 
                                        contents=input, 
                                        config=types.GenerateContentConfig(
                                                thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                ),)
    elif model_name == 'gemini-2.0-flash':
        return vertex_model.models.generate_content(model="gemini-2.0-flash", 
                                        contents=input)


def first_time_extraction(config, time_text, _r, vertex_model, adm_period, dis_period, whole_range, time_summary):
    def extract_answer(final_time_result, time_output):
        if 'Answer: ' in time_output:
            final_time_result += time_output.split('Answer: ')[1].strip()
        elif '[Answer]' in time_output:
            final_time_result += time_output.split('[Answer]')[1].strip()
        else:
            final_time_result += time_output.strip()
        return final_time_result
    
    time_question_prompt=open_file(config["prompts"]["time_simple_qs"]).replace("<<<CLinical Note>>>",time_text).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<Line_number>>>",_r[0])
    time_output = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt)
    time_output_first_text = time_output.text
    if 'Answer: ' in time_output_first_text:
        time_first_output = time_output_first_text.split('Answer: ')[1].strip()
    elif '[Answer]' in time_output_first_text:
        time_first_output = time_output_first_text.split('[Answer]')[1].strip()

    final_time_result = ""
    time_related_info = 'None'
    if 'Related Time Expression:' in time_output_first_text:
        time_related_info = time_output_first_text.split('Related Time Expression: ')[1].split('\n')[0].strip()
    
    if '1' in time_first_output:
        final_time_result += time_related_info
    if '2' in time_first_output:
        time_question_prompt=open_file(config["prompts"]["time_structured"]).replace('<<FIRST_OUTPUT>>', time_related_info).replace('<<<TEMPORTAL_EXPRESSION>>>', time_summary).replace("<<<CLinical Note>>>",time_text).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<Line_number>>>",_r[0])
        time_output = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt).text
        final_time_result = extract_answer(final_time_result, time_output)
    if '3' in time_first_output:
        if 'adm' in time_related_info.lower():
            final_time_result += adm_period
        elif 'dis' in time_related_info.lower():
            final_time_result += dis_period
        elif 'yester' in time_related_info.lower():
            pass
    if '4' in time_first_output:
        time_question_prompt=open_file(config["prompts"]["time_structured"]).replace('<<FIRST_OUTPUT>>', time_related_info).replace('<<<TEMPORTAL_EXPRESSION>>>', time_summary).replace("<<<CLinical Note>>>",time_text).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<Line_number>>>",_r[0])
        time_output = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt).text
        if 'Answer: ' in time_output:
            time_output = time_output.split('Answer: ')[1].strip()
        elif '[Answer]' in time_output:
            time_output = time_output.split('[Answer]')[1].strip()
        else:
            time_output = time_output.strip()
        final_time_result += get_day_before_and_after(time_output)
        
    if '5' in time_first_output:
        time_question_prompt=open_file(config["prompts"]["time_hodpod_between"]).replace('<<FIRST_OUTPUT>>', time_related_info).replace('<<<TEMPORTAL_EXPRESSION>>>', time_summary).replace("<<<CLinical Note>>>",time_text).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<Line_number>>>",_r[0])
        time_output = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt).text
        final_time_result = extract_answer(final_time_result, time_output)
    if '6' in time_first_output:
        final_time_result += whole_range
    
    return time_related_info, time_first_output, time_output_first_text, final_time_result

def second_time_processing(config, time_text, _r, vertex_model, time_summary, final_time_result):
    time_question_prompt=open_file(config["prompts"]["time_cls"]).replace("<<<CLinical Note>>>",time_text).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<Line_number>>>",_r[0])
    time_output = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt)
    time_output_text = time_output.text
    if 'Answer: ' in time_output_text:
        time_second_output = time_output_text.split('Answer: ')[1].strip()
    elif '[Answer]' in time_output_text:
        time_second_output = time_output_text.split('[Answer]')[1].strip()
    else:
        time_second_output = time_output_text

    if '1' in time_second_output:
        time_question_prompt=open_file(config["prompts"]["time_n_days"]).replace('<<<TEMPORTAL_EXPRESSION>>>', time_summary).replace("<<<CLinical Note>>>",time_text).replace("<<<ENTITY>>>",re.search(r'\{([^:]+):', _r[3]).group(1)).replace("<<<Line_number>>>",_r[0])
        time_output = gemini_inference(vertex_model, "gemini-2.5-flash", time_question_prompt).text
        if '0 day' != time_output.strip() and '0 days' != time_output.strip():
            final_time_result += ' for ' + time_output.strip()
    else:                                
        if '2' in time_second_output:
            final_time_result = "end at " + final_time_result

    return time_second_output, time_output_text, final_time_result