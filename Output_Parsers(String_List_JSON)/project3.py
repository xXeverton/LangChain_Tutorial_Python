from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


load_dotenv()


model = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=1000,
    verbose=True,
)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()
    chain = prompt | model | parser

    return  chain.invoke({
        "input": "dog"
    })


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a list of 10 synonyms for the following word. Return the result as a comma seperated list."),
        ("human", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser

    return chain.invoke({
        "input": "happy"
    })


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        recipe: str = Field(description="The name of the recepie")
        ingredients: list = Field(description="ingredients")
        
    

    parser = JsonOutputParser(pydantic_object=Person)
    chain = prompt | model | parser

    return chain.invoke({
        "phrase": "The ingredients for Margherita pizza are tomatoes, onions, chesse, bacon",
        "format_instructions": parser.get_format_instructions()
    })


   
# print(call_string_output_parser())
# print(call_list_output_parser())
print(call_json_output_parser())