# This code has been created based on the below
# : https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/#create-agent-supervisor
import functools
import operator
import os
from typing import List, Sequence, TypedDict, Annotated

from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic.v1 import BaseModel

# Getting the Open AI API key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print("In tool 1 ", a)
    print("In tool 1 ", b)
    return a * b


@tool
def add(a: int, b: int) -> int:
    """add two numbers."""
    print("In tool 2 ", a)
    print("In tool 2 ", b)
    return a + b

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


class SupervisorAgency(BaseModel):
    members: List = ["multiplication_node", "addition_node"]

    def create_agent(self, llm, tools, system_prompt: str):
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
        return executor

    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        # print("name ", name)
        # print("state ", state)
        # print("result ", result)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}

    def create_supervisor_agent(self):
        options = ["FINISH"] + self.members
        members=self.members
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers:  {members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
        )
        # Our team supervisor is an LLM node. It just picks the next agent to process
        # and decides when the work is completed
        options = ["FINISH"] + self.members
        # Using openai function calling can make output parsing easier for us
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ],
                    }
                },
                "required": ["next"],
            },
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(options), members=", ".join(self.members))

        llm = ChatOpenAI(model="gpt-4-1106-preview")

        supervisor_chain = (
                prompt
                | llm.bind_functions(functions=[function_def], function_call="route")
                | JsonOutputFunctionsParser()
        )

        return supervisor_chain


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4-1106-preview")
    sa = SupervisorAgency()
    multiplication_agent = sa.create_agent(llm, [multiply],
                                     "You are an expert in multiplication. Given two numbers,you multiply them and "
                                     "return the result. Please only do multiplication operation.")

    multiplication_node = functools.partial(sa.agent_node, agent=multiplication_agent, name="multiplication_node")
    addition_agent = sa.create_agent(llm, [add], "You are an expert in addition. Given two numbers,you add them and "
                                               "return the result. Please only do addition operation.")

    addition_node = functools.partial(sa.agent_node, agent=addition_agent, name="addition_node")
    workflow = StateGraph(AgentState)
    workflow.add_node("multiplication_node", multiplication_node)
    workflow.add_node("addition_node", addition_node)
    workflow.add_node("supervisor", sa.create_supervisor_agent())
    for member in sa.members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in sa.members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    workflow.add_edge(START, "supervisor")

    graph = workflow.compile()
    # graph.get_graph().print_ascii()

    for s in graph.stream(
            {
                "messages": [
                    HumanMessage(content="Please add 5 and 6.")
                ]
            }
    ):
        if "__end__" not in s:
            print(s)
            print("----")
