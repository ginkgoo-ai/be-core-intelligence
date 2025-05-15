import os
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
from langchain_core.globals import set_debug
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_verbose
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig

import threading

set_llm_cache(InMemoryCache())
set_debug(True)
set_verbose(True)


global_static_map = {}

def update_map(key, value):
    global global_static_map
    global_static_map[key] = value

def get_map():
    return global_static_map

thread_local = threading.local()

@tool(description="The information currently needed to fill the form")
def search(config: RunnableConfig):
    """
        The information currently needed to fill the form

    """
    thread_id = config["configurable"]["thread_id"]

    if thread_id:
        return global_static_map[thread_id]


tools = [search]

tools_node = ToolNode(tools)

system_message = ChatPromptTemplate.from_messages(
    [SystemMessage(
        content=("""
                # Role 
                    You are a backend of a Google plugin, analyzing the html web pages sent from the front end, 
                    analyzing the form items that need to be filled in them, retrieving the customer data that has been provided, 
                    filling in the form content, and returning it to the front end in a fixed json format. only json information is needed

                # Response json format
                    {
                      "actions": [
                        {
                          "selector": "input[id="password"][type="password"][name="password"]",
                          "type": "input",
                          "value": "qqqqqqqqqqqq"
                        },
                        {
                          "selector": "#currentAddressFiveYears input[id="value_true"][type="radio"]",
                          "type": "click"
                        },
                        {
                          "selector": "#citizenshipCeremony select[id="ceremonyCouncil"]",
                          "type": "input",
                          "value": "CORP_CITY_OF_LONDON"
                        },
                        {
                          "selector": "input[id="submit"][type="submit"][name="submit"]",
                          "type": "click"
                        }
                      ]
                    }
            """)
    ),
        HumanMessagePromptTemplate.from_template("{messages}"),

    ]
)

model = system_message | ChatOpenAI(
    base_url=os.getenv(os.getenv("MODEL_BASE_URL")),
    api_key=SecretStr(os.getenv("MODEL_API_KEY")),
    model=os.getenv(os.getenv("MODEL_WORKFLOW_NAME")),
    temperature=0,
).bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"

    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tools_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

