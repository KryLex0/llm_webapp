import json
from langchain_ollama import OllamaEmbeddings
from devtools import debug

# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import BaseTool, tool
from typing import cast
from langchain.schema.language_model import BaseLanguageModel
from IPython.display import Markdown

# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer, AutoModelForCausalLM

import requests

api_url: str = "http://localhost:11434"

def ask_model(question: str, llm_model: BaseLanguageModel):
    """
    Pose une question au modèle TinyLlama via l'API locale d'Ollama.

    Args:
        question (str): La question à poser au modèle.
        model_name (str): Le nom du modèle à utiliser (par défaut: "tinyllama").
        api_url (str): L'URL de l'API Ollama (par défaut: "http://localhost:11434").

    Returns:
        str: La réponse du modèle.
    """

    # replace _ by .
    try:
        response = llm_model.invoke(question)
        return response.content
        # # Préparer la requête
        # payload = {"model": model_name, "prompt": question}
        # headers = {"Content-Type": "application/json"}

        # # Envoyer la requête POST à l'API Ollama
        # response = requests.post(f"{api_url}/api/generate", json=payload, headers=headers)

        # # Vérifier le statut de la réponse
        # if response.status_code == 200:
        #     # Debug print the response text
        #     print("Response text:", response.text)
            
        #     # Split the response text into individual JSON objects
        #     json_objects = response.text.strip().split('\n')
        #     full_response = ""
        #     for obj in json_objects:
        #         data = json.loads(obj)
        #         full_response += data.get("response", "")
            
        #     return full_response if full_response else "Aucune réponse reçue."
        # else:
        #     return f"Erreur API ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Erreur lors de l'interrogation du modèle : {e}"

def get_llm_models(url: str = "http://localhost:11434"):
    """
    List all available LLM models from the Ollama API.
    """
    all_llm_models = {}
    try:
        response = requests.get(f"{url}/api/tags")
        if response.status_code == 200:
            json_models = response.json()
            all_llm_models = [model['model'] for model in json_models.get('models', [])]
        else:
            print(f"Erreur API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles LLM : {e}")

    return all_llm_models

@tool
def model_create_tic_tac_toe_board(boardgame: list):
    """
    Return the number of cells already filled in the boardgame.

    Returns:
        int: The number of cells already filled in the boardgame.
    """

    nb_filled_cells = 0

    for row in boardgame:
        for cell in row:
            if cell != "-":
                nb_filled_cells += 1

    return nb_filled_cells

@tool
def get_ttt_position(boardgame: list, player_form: str):
    """
    Return the position of the move in the Tic Tac Toe game as "X,Y" coordinates.
    The coordinates are based on the boardgame length.

    Returns:

    """

    X = 0
    Y = 0

    board_max = len(boardgame)
    if X < 0 or X > board_max or Y < 0 or Y > board_max:
        return "Invalid position"

    new_position = f"{X},{Y}"

    return new_position

def create_basemodel(llm_model: str, tools: list):
    llm = ChatOllama(
        model=llm_model,
        base_url="http://localhost:11434",
    )

    llm_basemodel = cast(BaseLanguageModel, llm.bind()) #response_format={"type": "json_object"}))
    llm_basemodel.bind_tools(tools)
    return llm_basemodel


def model_play_tic_tac_toe(llm_model: BaseLanguageModel, boardgame: list, player_form: str, invalid_position: str, board_max: int):
    # store the boardgame as a string like cel1,cel2,cel3;cel4,cel5,cel6;cel7,cel8,cel9
    boardgame_str = ";".join([",".join(row) for row in boardgame])

    # boardgame_str_formatted = ""


    context = f"You are playing a game of tic-tac-toe {board_max}x{board_max}. The board is as follows:\n{boardgame_str}."
    context += f"\nIt is your turn, Player {player_form}. Return no text, only the line and column where you placed your move, separated by a comma. \
        For example, to place an '{player_form}' in the top left corner, return '1,1'. You can't place a move in a cell that is already occupied. \
        Integers can only be between 1 and 3. \
        Te result i'm expecting is 'X,Y' ONLY. \
        Invalid positions are:\n{invalid_position}."

    # context = "It is your turn, Player O. Place an 'O' on the board by selecting a cell currently marked with '-'. Return only the line and column numbers of your move, separated by a comma (e.g., '1,1' for the top-left corner). Do not place your move in an already occupied cell. Valid integers for rows and columns range from 1 to 3 inclusive."
  
    #Return only the board, no text is needed."

    # context = "I want you to act as a Tic-Tac-Toe game. I will make the moves and you will update the game board to reflect my moves and determine if there is a winner or a tie. Use X for my moves and O for the computer's moves. Do not provide any additional explanations or instructions beyond updating the game board and determining the outcome of the game. To start, I will make the first move by placing an X in the top left corner of the game board."
    # context += "The board is as follows:\n" + boardgame_str

    # clear all spaces and new lines
    position = ask_model(context, llm_model)
    position = position.replace(" ", "").replace("\n", "")

    return position


@tool
def get_json(json_url: str):
    """
    Function to retrieve all jobs data from the JSON

    Returns:
    JSON data
    """
    import requests

    content = requests.get(json_url).json()
    return content


def ask_json_model(question: str, llm_model: BaseLanguageModel, tools: list, json_url: str):
    # https://gist.githubusercontent.com/svngoku/dadcd4ddd28ed52e70e20c6aa3b81b72/raw/3f608edbad8b3316af303e45a6f94816db76ab9d/jobs.json
    # JSON_URL = "https://gist.githubusercontent.com/svngoku/dadcd4ddd28ed52e70e20c6aa3b81b72/raw/3f608edbad8b3316af303e45a6f94816db76ab9d/jobs.json"

    # tools = [get_json]
    # llm_model = create_basemodel(llm_name, tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful Job Assistant. Please use the provided tool"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm_model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    # Construire l'entrée pour inclure json_url
    input_data = {
        "input": f"{question}. The JSON URL is {json_url}",
    }
    # response = agent_executor.invoke({"input": question, "json_url": json_url})

    # Exécuter l'agent
    response = agent_executor.invoke(input_data)
    # Markdown(response["output"])
    return response["output"]

    # r_remote = agent_executor.invoke({"input": question})
    # # debug(r)
    # print("Remote Jobs")
    # Markdown(r_remote["output"])

    # r_remote = agent_executor.invoke({"input": "I want to get the most skills required job"})
    # # debug(r)
    # print("Remote Jobs")
    # Markdown(r_remote["output"])

def main():
  all_llm_models = get_llm_models()
  print("Modèles LLM disponibles:", all_llm_models)
  question = "What is the capital of France?"
#   llm_model1 = create_basemodel(all_llm_models[0], [])
#   response1 = ask_model(question, llm_model1)
# #   llm_model2 = create_basemodel(all_llm_models[1], [])
# #   response2 = ask_model(question, llm_model2)
#   print("Réponse du modèle:", response1)

#   llm_name = "tinyllama:latest"
#   llm_name = "llama3.2:1b"
  llm_name = "llama3.2:latest"

  JSON_URL = "https://gist.githubusercontent.com/svngoku/dadcd4ddd28ed52e70e20c6aa3b81b72/raw/3f608edbad8b3316af303e45a6f94816db76ab9d/jobs.json"
  # question = "I want to get the most skills required job"
  question = "How much of the jobs are in departement of Ile-de-France?"

  tools = [get_json]
  llm_model = create_basemodel(llm_name, tools)

  response = ask_json_model(question=question, llm_model=llm_model, tools=tools, json_url=JSON_URL)
  print("Question:", question)
  print("Réponse du modèle:")
  Markdown(response)




if __name__ == "__main__":
  main()