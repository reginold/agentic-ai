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
def python_repl_tool(code: Annotated[str, "Python code to execute for generating a chart."]):
    """
    Executes Python code in a REPL environment. 
    Checks if 'output.png' is created after execution.
    """
    try:
        print(f"Executing code:\n{code}")
        result = repl.run(code)
        if "plt.savefig" in code:  # Ensure code contains the savefig call
            import os
            if os.path.exists("output.png"):
                return f"Execution succeeded and chart saved as 'output.png':\n{result}"
            else:
                return f"Execution succeeded but 'output.png' was not created. Result:\n{result}"
        return f"Execution succeeded:\n{result}"
    except Exception as e:
        return f"Execution failed. Error: {repr(e)}"



# Helper Functions
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants. "
        "Use the provided tools to make progress. If you cannot complete the task, "
        "leave it for another assistant. Prefix 'FINAL ANSWER' if the task is complete. "
        f"{suffix}"
    )

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

# Agents
llm = ChatOpenAI(model="Meta-Llama-3.1-8B-Instruct")

research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    state_modifier=make_system_prompt("You can only perform research."),
)

chart_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    state_modifier=make_system_prompt(
        "You can only generate charts using Python. Generate Python code that uses matplotlib "
        "to create a chart and saves it as 'output.png'. Include all necessary imports."
    ),
)


# Nodes
def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    last_message = result["messages"][-1]
    print(f"Research Agent Result: {last_message.content}")
    goto = get_next_node(last_message, "chart_generator")
    result["messages"][-1] = HumanMessage(content=last_message.content, name="researcher")
    return Command(update={"messages": result["messages"]}, goto=goto)

def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    last_message = result["messages"][-1]
    print(f"Chart Agent Result: {last_message.content}")
    goto = get_next_node(last_message, "researcher")
    result["messages"][-1] = HumanMessage(content=last_message.content, name="chart_generator")
    return Command(update={"messages": result["messages"]}, goto=goto)

# Graph Workflow
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_edge(START, "researcher")
graph = workflow.compile()

# Invocation
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Get the UK's GDP over the past 5 years and create a line chart of it.",
            }
        ]
    },
    {"recursion_limit": 150},
)

for event in events:
    print(event)
    print("----")
