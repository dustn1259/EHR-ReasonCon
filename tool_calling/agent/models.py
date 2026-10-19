import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types

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


def get_azure_response(client, prompt, model, system_prompt="You are a helpful medical assistant analyzing clinical notes."):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=model,
    )
    return response.choices[0].message.content
