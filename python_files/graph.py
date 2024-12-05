import os
import sys
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

# create llm using chatOllama with tinyllama model
llm = ChatOllama(model_name="tinyllama")

llm_transformer = LLMGraphTransformer(llm=llm)

text = "Capital of France is Paris and is situated in Europe."

documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes: {graph_documents[0].nodes}")
print(f"Relationships: {graph_documents[0].relationships}")