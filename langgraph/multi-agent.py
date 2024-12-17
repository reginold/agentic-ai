# import openai
# from langsmith import wrappers, traceable

# # Auto-trace LLM calls in-context
# client = wrappers.wrap_openai(openai.Client())

# @traceable # Auto-trace this function
# def pipeline(user_input: str):
#     result = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "Answer the question in a couple sentences."},
#             {"role": "user", "content": user_input}
#         ],
#         model="Meta-Llama-3.1-8B-Instruct"
#     )
#     print(result.choices[0].message.content)
#     return result.choices[0].message.content

# pipeline("Share a happy story with me")
# # Out:  Once upon a time, in a small village nestled among rolling hills, there lived a young girl named Lily. Every day, Lily would venture into the forest, where she would spend hours playing with the colorful butterflies and listening to the gentle rustling of the leaves. One sunny afternoon, as she was sitting on a mossy rock, she found a tiny, injured bird. Without hesitation, Lily took the bird home and nursed it back to health. The bird, grateful for Lily's kindness, would visit her every day, bringing her joy and reminding her of the beauty of the world. Lily's heart was filled with happiness, knowing that she had made a new friend and brought a little bit of magic into her life.

# Import necessary libraries
from typing import Annotated, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

# Tools
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )



# Helper Functions
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

# Agents
llm = ChatOpenAI(model="Meta-Llama-3.1-405B-Instruct")


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# Research agent and node
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    state_modifier=make_system_prompt(
        "You can only perform research and output results as JSON. "
        "Do not include any explanations or additional text. "
        "Provide the data in the following format: "
        '{"data": [{"year": 2018, "gdp": 2.13}, {"year": 2019, "gdp": 2.17}, {"year": 2020, "gdp": 2.06}, {"year": 2021, "gdp": 8.67}, {"year": 2022, "gdp": 4.35}]}'
    ),
)


import re
import json

def extract_json(text: str) -> str:
    """
    Extract the first valid JSON object from a given text.
    Handles cases where the model outputs non-JSON text along with JSON.
    """
    json_pattern = r'(\{.*\})'  # Match JSON starting with '{' and ending with '}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        return matches[0]  # Return the first matched JSON block
    return "{}"  # Return empty JSON if no match is found

def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    
    # Extract clean JSON content
    last_content = result["messages"][-1].content
    clean_json = extract_json(last_content)

    # Validate JSON and handle different structures
    try:
        json_output = json.loads(clean_json)

        # Check if 'data' exists; otherwise, assume the JSON is the data itself
        if "data" in json_output:
            data = json_output["data"]
        else:
            # Assume json_output is directly the data
            data = json_output

        # Wrap the extracted data into Python code for the REPL tool
        wrapped_output = {
            "name": "python_repl_tool",
            "parameters": {"code": f"data = {json.dumps(data)}\nprint(data)"}
        }
    except json.JSONDecodeError:
        wrapped_output = {"error": "Invalid JSON format extracted from model output"}

    # Update the message with the wrapped function call
    result["messages"][-1] = HumanMessage(
        content=json.dumps(wrapped_output), name="researcher"
    )

    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )


# Chart generator agent and node
# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    state_modifier=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague."
    ),
)

def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

# Graph Workflow
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "researcher")
graph = workflow.compile()

events = graph.stream(
    {
        "messages": [
            (
                "user",
                "First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, finish.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print(s)
    print("----")
