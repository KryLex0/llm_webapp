import os
import sys
import uuid
from typing import Annotated

from devtools import debug

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from python_files.model import get_llm_obj
from langchain_chroma import Chroma

class Stater(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        """
        Initialize the Assistant with a runnable.
        """
        self.runnable = runnable

@tool
def retrieve_documents(query: str) -> list:
    """Retrieve documents from the vector store based on the query."""
    return retriever.invoke(query)


@tool
def generate_answer(answer: str) -> str:
    """You are an assistant for question-answering tasks.
    Use the retrieved documents to answer the user question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

    return f"Here is the answer to the user question: {answer}"


@tool
def grade_document_retrieval(step_by_step_reasoning: str, score: int) -> str:
    """You are a teacher grading a quiz. You will be given:
    1/ a QUESTION
    2/ a set of comma separated FACTS provided by the student
    """
    return f"Grading result: {score}"


@tool
def basic_web_search(query: str) -> str:
    """Run web search on the question.  Call Tivaly if we have a key, DuckDucGo otherwise"""

    if os.environ.get("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily_tool = TavilySearchResults(max_results=5)
        docs = tavily_tool.invoke({"query": query})
        web_results = "\n".join([d["content"] for d in docs])
    else:
        from langchain_community.tools import DuckDuckGoSearchRun

        duckduck_search_tool = DuckDuckGoSearchRun()
        web_results = duckduck_search_tool.invoke(query)

    return web_results


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# use embeddings from OpenAI
embeddings = OpenAIEmbeddings(
    api_key="noneed",
    model="llama3.2:1b",
    base_url="http://127.0.0.1:11434/api"
)


try:
    from langchain_core.vectorstores import InMemoryVectorStore

    vectorstore = InMemoryVectorStore(embeddings)

except Exception as e:
    print(f"Error creating vector store: {e}")
    sys.exit(1)


retriever = vectorstore.as_retriever(k=4)

# Example usage
response = retrieve_documents.invoke("agent risks")
print("Retriever response:", response)