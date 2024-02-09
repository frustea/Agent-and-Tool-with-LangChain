import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.runnable import RunnableLambda


### Tagging function

class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")


model = ChatOpenAI(temperature=0)
tagging_functions = [convert_pydantic_to_openai_function(Tagging)]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)


tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

tagging_chain.invoke({"input": "wait! what did you say?!"})

### Extraction function 

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.preposterousuniverse.com/blog/2021/11/25/thanksgiving-16/")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]


class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")

overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]
tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()


### Example from the LangChain templates:
class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]

paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]
extraction_model = model.bind(
    functions=paper_extraction_function, 
    function_call={"name":"Info"}
)

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])


extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
splits = text_splitter.split_text(doc.page_content)
def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list
prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
)
chain = prep | extraction_chain.map() | flatten
chain.invoke(doc.page_content)