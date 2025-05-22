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
    base_url=os.getenv("MODEL_BASE_URL"),
    api_key=SecretStr(os.getenv("MODEL_API_KEY")),
    model=os.getenv("MODEL_WORKFLOW_NAME"),
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


def check_node(state: MessagesState):
    """
    Use LLM to determine whether fill_data can satisfy the form requirements in the message.
    fill_data is obtained from the tools node (ToolNode) and placed in the state, the return value of the search tool is fill_data.
    """
    messages = state['messages']
    # The output of ToolNode will be placed in the state with the tool name as the key
    fill_data = state.get('messages')[-1].content
    prompt = f"""
        You are a form auto-filling assistant. prefer use english to filling 
        The current form requirements are as follows: {messages}
        The user-provided data is as follows: {fill_data}
        Please determine whether this data can satisfy the form filling requirements, or if the form can be filled automatically without additional data. If yes, return ok, otherwise return interrupt.
        Only return ok or interrupt.
    """
    response = model.invoke([{"role": "user", "content": prompt}])
    result = response.content.strip().lower()
    if "ok" in result:
        return {"check_result": "ok", "messages": messages, "search": fill_data}
    else:
        return {"check_result": "interrupt", "messages": messages, "search": fill_data}


def interrupt_node(state: MessagesState):
    """
    The interrupt node directly returns a prompt for manual input.
    """
    return {"messages": [{
        "type": "system",
        "content": """
            {"manual":"Unable to automatically fill all form items, please supplement manually."}
        """
    }]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tools_node)
workflow.add_node("check", check_node)
workflow.add_node("interrupt", interrupt_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "check")
workflow.add_conditional_edges(
    "check",
    lambda state: state["check_result"],
    {
        "ok": "agent",
        "interrupt": "interrupt"
    }
)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

