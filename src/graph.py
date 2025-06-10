import os
from typing import Literal

from langchain_core.messages import SystemMessage, BaseMessage
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
from typing import TypedDict, List, Optional

import threading

set_llm_cache(InMemoryCache())
set_debug(True)
set_verbose(True)


global_static_map = {}

class GraphState(TypedDict):
    messages:Optional[List[BaseMessage]]


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


def should_continue(state: GraphState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"

    return END


def call_model(state: GraphState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


def check_node(state: GraphState):
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


def interrupt_node(state: GraphState):
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

# ========== Q&A旁路记录功能 BEGIN ==========
qa_storage = []

def extract_questions_node(state: GraphState):
    """
    用大模型解析HTML，提取所有需要填写的表单项及其问题描述
    """
    import json
    messages = state['messages']
    html = ""
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            html = getattr(msg, "content", "")
            break
    prompt = """
    你是一个表单解析助手。请从以下HTML页面代码中，找出所有需要用户填写的表单项（如input、select、textarea），
    并为每个表单项生成一个简洁的"问题"描述（可用label、placeholder、aria-label、name、id等信息），
    以及该表单项的唯一CSS选择器（如#id、[name="xxx"]等）。
    返回格式为JSON数组，每个元素包含"selector"和"question"字段，例如：[{"selector": "username", "question": "请输入用户名"},  {"selector": "password", "question": "请输入密码"}]
    下面是HTML代码：{html}
    """.replace("{html}", html)

    response = model.invoke([{"role": "human", "content": prompt}])
    try:
        fields = json.loads(response.content)
    except Exception:
        fields = []
    return {"fields": fields, "messages": messages}

def match_answers_node(state: GraphState):
    """
    匹配AI生成的actions和fields，生成结构化Q&A
    """
    import json
    fields = state.get("fields", [])
    messages = state.get("messages", [])
    ai_reply = ""
    for msg in messages[::-1]:
        if getattr(msg, "type", None) == "ai":
            ai_reply = getattr(msg, "content", "")
            break
    try:
        actions = json.loads(ai_reply).get("actions", [])
    except Exception:
        actions = []
    qa_list = []
    for field in fields:
        selector = field["selector"]
        question = field["question"]
        answer = ""
        for act in actions:
            act_selector = act.get("selector", "")
            if selector in act_selector:
                if act.get("type") == "click":
                    answer = "click"
                elif "value" in act:
                    answer = act["value"]
                break
        qa_list.append({"question": question, "answer": answer})
    global qa_storage
    qa_storage.append(qa_list)
    print("Q&A记录：", qa_list)
    return {"qa_list": qa_list, "messages": messages}

# 注册旁路节点（不影响主流程）
workflow.add_node("extract_questions", extract_questions_node)
workflow.add_node("match_answers", match_answers_node)
workflow.add_edge("extract_questions", "match_answers")

# ========== Q&A旁路记录功能 END ==========

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

# ========== Q&A旁路记录入口函数 ==========
def run_with_sidecar(input_state, *args, **kwargs):
    """
    主流程invoke时自动并行记录Q&A，不影响主流程返回。
    合并输入和输出的messages，保证旁路节点能同时拿到原始HTML和AI回复。
    """
    result = app.invoke(input_state, *args, **kwargs)
    try:
        # 合并输入和输出的messages
        merged_state = {
            "messages": input_state["messages"] + result["messages"]
        }
        state1 = extract_questions_node(merged_state)
        match_answers_node(state1)
    except Exception as e:
        print("Q&A旁路记录异常：", e)
    return result
# ========== Q&A旁路记录入口函数 END ==========

