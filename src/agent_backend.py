import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get the current file's directory (src)
current_dir = Path(__file__).resolve().parent
# Look for .env in the parent directory (Construction Sidekick Agent)
env_path = current_dir.parent / '.env'

# Load it explicitly
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

# Debug check
if not api_key:
    print("ERROR: GROQ_API_KEY not found.")
    print(f"   Searching for .env at: {env_path}")
    print("   Please check your .env file exists and has the key.")
    sys.exit(1)

# IMPORTS
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# SETUP LLM
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
    api_key=api_key
)

# DEFINE TOOLS
@tool
def python_calculator(code: str):
    """
    A Python shell. Use this to execute python commands. 
    Input should be a valid python command. 
    Use this for ALL math calculations or data filtering.
    """
    repl = PythonREPL()
    try:
        result = repl.run(code)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# DEFINE STATE
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    context_data: str

# DEFINE NODES
def agent_reasoning_node(state: AgentState):
    messages = state['messages']
    context = state['context_data']

    system_prompt = (
        "You are a Construction Sidekick Agent for Sprout Solutions. "
        "You are an expert in reading Bill of Materials (BOM) and BOQ. \n\n"
        "CONTEXT FROM UPLOADED FILE:\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        "1. When asked to list materials for a specific area (e.g., 'Pantry'), you must list EVERY item found under that section header.\n"
        "2. INCLUDE items that are dimension (e.g., '1.65m x 0.60 Granite Counter'). These are materials.\n"
        "3. INCLUDE items that are descriptions of work (e.g., 'Replacement of Kitchen Cabinets').\n"
        "4. Do not leave out an item just because it lacks a specific quantity (e.g.,'Lot' or 'Length').\n"
        "5. if the user ask for a calculation, output a Python code block.\n"
        "6. If the user ask for a list, format it as a clean Markdown list"
    )
    
    full_history = [SystemMessage(content=system_prompt)] + messages
    llm_with_tools = llm.bind_tools([python_calculator])
    response = llm_with_tools.invoke(full_history)
    return {"messages": [response]}

def tool_execution_node(state: AgentState):
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls
    
    results = []
    for t in tool_calls:
        if t['name'] == 'python_calculator':
            print(f"   Running Tool: {t['args']['code']}")
            output = python_calculator.invoke(t['args'])
            results.append(AIMessage(content=f"Tool Output: {output}"))
    
    return {"messages": results}

# DEFINE GRAPH
def router(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_reasoning_node)
workflow.add_node("tools", tool_execution_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", router, {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()