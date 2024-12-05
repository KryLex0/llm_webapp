import json
from langchain_ollama import OllamaEmbeddings
from devtools import debug

# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_ollama import ChatOllama
from typing import cast

from IPython.display import Markdown

from transformers import AutoTokenizer, AutoModelForCausalLM
import requests


def ask_model(
    question: str,
    model_name: str = "tinyllama",
    api_url: str = "http://localhost:11434",
):
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
    model_name = model_name.replace("_", ".")
    try:
        # Préparer la requête
        payload = {"model": model_name, "prompt": question}
        headers = {"Content-Type": "application/json"}

        # Envoyer la requête POST à l'API Ollama
        response = requests.post(
            f"{api_url}/api/generate", json=payload, headers=headers
        )

        # Vérifier le statut de la réponse
        if response.status_code == 200:
            # Debug print the response text
            print("Response text:", response.text)

            # Split the response text into individual JSON objects
            json_objects = response.text.strip().split("\n")
            full_response = ""
            for obj in json_objects:
                data = json.loads(obj)
                full_response += data.get("response", "")

            return full_response if full_response else "Aucune réponse reçue."
        else:
            return f"Erreur API ({response.status_code}): {response.text}"
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
            all_llm_models = [model["model"] for model in json_models.get("models", [])]
        else:
            print(f"Erreur API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles LLM : {e}")

    return all_llm_models


@tool
def get_jobs(json_url: str):
    """
    Function to retrieve all jobs data from the JSON

    Returns:
    JSON data
    """
    import requests

    content = requests.get(json_url).json()
    return content


def ask_json_model(question: str, llm_name: str, json_url: str):
    # https://gist.githubusercontent.com/svngoku/dadcd4ddd28ed52e70e20c6aa3b81b72/raw/3f608edbad8b3316af303e45a6f94816db76ab9d/jobs.json
    # JSON_URL = "https://gist.githubusercontent.com/svngoku/dadcd4ddd28ed52e70e20c6aa3b81b72/raw/3f608edbad8b3316af303e45a6f94816db76ab9d/jobs.json"

    tools = [get_jobs]
    llm_model = get_llm_obj(llm_name)

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

def get_llm_obj(model_name: str):
    llm = ChatOllama(model=model_name)
    llm = cast(BaseLanguageModel, llm.bind(response_format={"type": "json_object"}))
    return llm

def main():
    all_llm_models = get_llm_models()
    print("Modèles LLM disponibles:", all_llm_models)
    # question = "What is the capital of France?"
    # response = ask_model(question, model_name=all_llm_models[0])
    # response = ask_model(question, model_name=all_llm_models[1])
    # print("Réponse du modèle:", response)

    # llm = get_llm_obj("tinyllama")
    # llm = get_llm_obj("llama3.2:1b")
    llm_name = "llama3.2:1b"

    JSON_URL = "https://gist.githubusercontent.com/svngoku/dadcd4ddd28ed52e70e20c6aa3b81b72/raw/3f608edbad8b3316af303e45a6f94816db76ab9d/jobs.json"
    # question = "I want to get the most skills required job"
    question = "How much of the jobs are in departement of Ile-de-France?"

    response = ask_json_model(llm=llm_name, question=question, json_url=JSON_URL)
    print("Question:", question)
    print("Réponse du modèle:")
    Markdown(response)


if __name__ == "__main__":
    main()
