# In this pattern, the supervisor splis the task and
# assigns each task to follower agents
# finally the last node in the graph aggregates the answer
# and provides a consolidated answer
import ast
import operator
import os
from typing import Sequence, TypedDict, Annotated

from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph

# Getting the Open AI API key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4")


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    # next: str


@tool
def add(a: int, b: int) -> int:
    """add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def supervisor(state):
    print("state ", state)
    workers = [{"name": "multiplication_expert", "skill": "proficient in solving multiplication problems"},
               {"name": "addition_expert", "skill": "proficient in solving addition problems"}]
    system_prompt = """
    You are a supervisor tasked with assigning tasks to your workers.Given as task,
    you first split the task based on the skills your workers possess. Then you map each task to 
    the relevant worker. You have access to the following workers : {{workers}}. Please output the 
    workers mapped to their respective tasks in a JSON format. Here is an example of the output
    
    Example:
    Workers:
    [{{"name":"substraction_expert","skill":"proficient in solving substraction problems"}},
    {{"name":"division_expert","skill":"proficient in solving division problems"}}]
    
    Input: 
    Divide 50 by 10 and then subtract 10 from the result
    
    Output:
    {{"division_expert":"Divide 50 by 10","subtraction_expert":"Subtract 10 from result"}}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    agent = prompt | llm | JsonOutputParser()

    result = agent.invoke(state)
    print(result)
    # print("name ", name)
    # print("state ", state)
    # print("result ", result)
    return {"messages": [HumanMessage(content=str(result))]}


def addition_expert(state):
    message = [ast.literal_eval(state['messages'][-1].content)['addition_expert']]
    print("addition ", message)
    tools = [add]
    system_prompt = """
    You are an expert in addition. Given two numbers,you add them and return the result."
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    result = executor.invoke({"messages": message})
    return {"messages": [HumanMessage(content=result["output"])]}


def multiplication_expert(state):
    message = [ast.literal_eval(state['messages'][-1].content)['multiplication_expert']]
    print("addition ", message)
    tools = [multiply]
    system_prompt = """
    You are an expert in multiplication. Given two numbers,you multiply them and return the result.
    """


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    result = executor.invoke({"messages": message})
    return {"messages": [HumanMessage(content=result["output"])]}


def generate_final_answer(state):
    system_prompt = """
    You are responsible to provide the final answer based on the sequence of messages provided to you
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    agent = prompt | llm
    result = agent.invoke(state)
    print("final ", result)

    return {"messages": [HumanMessage(content=result.content)]}


workflow = StateGraph(AgentState)
workflow.add_node("supervisor_node", supervisor)
workflow.add_node("addition_expert", addition_expert)
workflow.add_node("multiplication_expert", multiplication_expert)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "supervisor_node")
workflow.add_edge("supervisor_node", "addition_expert")
workflow.add_edge("supervisor_node", "multiplication_expert")
workflow.add_edge(["addition_expert", "multiplication_expert"], "generate_final_answer")
graph = workflow.compile()
# graph.get_graph().print_ascii()
for s in graph.stream(
        {
            "messages": [
                HumanMessage(content="Please add 5 and 6 and multiply 5 by 10.")
            ]
        }
):
    if "__end__" not in s:
        print(s)
        print("----")
