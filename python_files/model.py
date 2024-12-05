import json
from langchain_ollama import OllamaEmbeddings
from devtools import debug

# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM

import requests
def ask_model(question: str, model_name: str = "tinyllama", api_url: str = "http://localhost:11434"):
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
        response = requests.post(f"{api_url}/api/generate", json=payload, headers=headers)

        # Vérifier le statut de la réponse
        if response.status_code == 200:
            # Debug print the response text
            print("Response text:", response.text)
            
            # Split the response text into individual JSON objects
            json_objects = response.text.strip().split('\n')
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
            all_llm_models = [model['model'] for model in json_models.get('models', [])]
        else:
            print(f"Erreur API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles LLM : {e}")

    return all_llm_models

def main():
  all_llm_models = get_llm_models()
  print("Modèles LLM disponibles:", all_llm_models)
  question = "What is the capital of France?"
  response = ask_model(question, model_name=all_llm_models[0])
  response = ask_model(question, model_name=all_llm_models[1])
  print("Réponse du modèle:", response)


if __name__ == "__main__":
  main()