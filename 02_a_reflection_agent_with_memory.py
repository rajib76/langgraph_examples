# Author: Rajib Deb
# This example shows a simple implementation of a reflection agent
import operator
import os
from typing import TypedDict, Annotated, Sequence, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAI, ChatOpenAI
from langgraph.constants import END
from langgraph.graph import add_messages, StateGraph, MessageGraph
from memory.semantic_memory import SemanticMemory
from pydantic.v1 import BaseModel

# Getting the Open AI API key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI()


# The shared state of the agents
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]



class ReflectionAgent(BaseModel):
    model: str = "gpt-3.5-turbo"

    # The first agent in the workflow that generates the answer
    def generate_answer(self, state: State):
        messages = state["messages"]
        generation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You will answer based on a question asked."
                " Please ensure to add all necessary details."
                "If you receive a critique, respond with a modified version of your response that incorporates the "
                "critique comments.",
            ),
            MessagesPlaceholder(variable_name="messages"), ])
        generate = generation_prompt | llm
        messages = generate.invoke({"messages": messages})
        return {"messages":[messages.content]}

    # The first agent in the workflow that critiques the answer
    def critique_answer(self, state: State):
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You will be grading a provided answer. Generate critique and recommendations for the user's "
                    "submission. "
                    " Provide detailed recommendations, including requests for length, depth, style, etc.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        reflect = reflection_prompt | llm
        original_question = state["messages"][0]
        answer = state["messages"][-1]
        # original_question = [HumanMessage(content=state[0].content)]
        # answer = [HumanMessage(content=state[-1].content)]
        messages = reflect.invoke({"messages": [original_question + answer]})
        return {"messages":[messages.content]}

    def add_to_memory(self, state: State):
        original_question = state["messages"][0]
        answer = state["messages"][-1]
        sc = SemanticMemory()
        semantic_val, memory_item = sc.add_memory_item(mem_key=str(original_question),
                                                       mem_val=str(answer))

        return {"messages":[memory_item]}

    def should_continue(self, state: List[BaseMessage]):
        if len(state["messages"]) > 8:
            # End after 4 iterations
            return "add_to_memory"
        return "critique_agent"

    def make_workflow(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("generate_agent", self.generate_answer)
        graph_builder.add_node("critique_agent", self.critique_answer)
        graph_builder.add_node("add_to_memory", self.add_to_memory)
        graph_builder.set_entry_point("generate_agent")
        graph_builder.add_conditional_edges("generate_agent", self.should_continue)
        graph_builder.add_edge("critique_agent", "generate_agent")
        graph_builder.set_finish_point("add_to_memory")

        graph = graph_builder.compile()

        return graph


if __name__ == "__main__":
    reflection_agent = ReflectionAgent()
    graph = reflection_agent.make_workflow()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        for event in graph.stream({"messages":[user_input]}):
            for key,value in event.items() :
                print(str(key) + ":" + str(value["messages"][0]))



