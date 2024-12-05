import os
import sys
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
# print("Loaded documents:", docs)

# Flatten the list of documents
docs_list = [item for sublist in docs for item in sublist]
# print("Flattened documents list:", docs_list)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
# print("Document splits:", doc_splits)

# use embeddings from OpenAI
embeddings = OpenAIEmbeddings(
    api_key="noneed",
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434/api"
)

try:
    doc_embeddings = embeddings.embed_documents([doc.page_content for doc in doc_splits])
    vectorstore = FAISS.from_documents(doc_embeddings, embeddings)
    print("Vector store created successfully.")
except Exception as e:
    print(f"Error creating vector store: {e}")
    sys.exit(1)

retriever = vectorstore.as_retriever(k=4)
print("Retriever created successfully.")

# Example usage
try:
    response = retriever.invoke("agent risks")
    print("Retriever response:", response)
except Exception as e:
    print(f"Error invoking retriever: {e}")