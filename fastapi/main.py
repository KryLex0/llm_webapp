import sys
import os
from typing import Annotated

# Add the parent directory of sample_project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Union
from fastapi import APIRouter, FastAPI, Request, Form
from python_files.model import ask_model, get_llm_models, ask_json_model
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

templates = Jinja2Templates(directory='templates')

app = FastAPI()
# router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/ask')
async def ask_main(request: Request):
    """
    Main page for asking questions to the LLM models.

    Args:
        request (Request): The request object.

    Returns:
        TemplateResponse: The response to the request.
    
    Passes the list of available LLM models to the template to be displayed in the HTML.
    """
    llm_models = get_llm_models()
    return templates.TemplateResponse('ask.html', {'request': request, 'llm_models': llm_models})

# @app.get('/ask?model={model}', response_class=HTMLResponse)
# async def ask_main_model(request: Request):
#     return templates.TemplateResponse('index.html', {'request': request})

@app.post("/ask/response")
async def post_question(question: Annotated[str, Form(...)], model: Annotated[str, Form(...)]):
    return model, ask_model(question, model)

@app.get("/search")
async def search_main(request: Request):
    """
    Main page for asking questions to the LLM models with URL containing json data. 

    Args:
        request (Request): The request object.

    Returns:
        TemplateResponse: The response to the request.
    
    Passes the list of available LLM models to the template to be displayed in the HTML.
    """

    llm_models = get_llm_models()
    return templates.TemplateResponse('search.html', {'request': request, 'llm_models': llm_models})

@app.post("/search/response")
async def post_search(question: Annotated[str, Form(...)], model: Annotated[str, Form(...)], json_url: Annotated[str, Form(...)]):
    return model, ask_json_model(question, model, json_url)