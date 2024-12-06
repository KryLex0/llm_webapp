import sys
import os
from typing import Annotated
from starlette.responses import StreamingResponse
import io

# Add the parent directory of sample_project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Request, Form
from python_files.model import ask_model, get_llm_models, ask_json_model, create_basemodel, get_json
from python_files.game import start_game
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
async def post_question(question: Annotated[str, Form(...)], llm_model_name: Annotated[str, Form(...)]):
    model = create_basemodel(llm_model_name, [])
    response = {
        "model_name": model.model,
        "question": question,
        "response": ask_model(question, model)
    }
    return response

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
async def post_search(question: Annotated[str, Form(...)], llm_model_name: Annotated[str, Form(...)], json_url: Annotated[str, Form(...)]):
    model = create_basemodel(llm_model_name, [])
    tools = [get_json]
    response = {
        "model_name": model.model,
        "question": question,

        "response": ask_json_model(question, model, tools, json_url)

    }
    return response

@app.get("/ttt")
async def ttt_main(request: Request):
    # Create a stream to capture the prints
    stream = io.StringIO()
    # Redirect stdout to the stream
    sys.stdout = stream

    # Call the function that performs the prints
    start_game()

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Get the content of the stream
    content = stream.getvalue()

    # Return the content as a streaming response
    return StreamingResponse(io.StringIO(content), media_type="text/plain")

# @app.get("/ttt")
# async def ttt_main(request: Request):
#     async def stream_output():
#         stream = io.StringIO()
#         # Capture stdout pour rediriger les print()
#         original_stdout = sys.stdout
#         sys.stdout = stream

#         try:
#             # Appeler la fonction start_game (asynchrone ou non)
#             await start_game(output_stream=stream)
#         finally:
#             # Réinitialiser stdout
#             sys.stdout = original_stdout

#         # Lire les données au fur et à mesure
#         while True:
#             output = stream.getvalue()
#             if output:
#                 yield output
#                 # Réinitialiser le buffer pour éviter les doublons
#                 stream.seek(0)
#                 stream.truncate(0)
#             else:
#                 await asyncio.sleep(0.1)

#     return StreamingResponse(stream_output(), media_type="text/plain")