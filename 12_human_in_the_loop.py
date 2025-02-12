import os
from typing import TypedDict, Annotated
import uuid
import openai  # OpenAI API for GPT-4o
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.types import interrupt, Command

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# **Set Up OpenAI Client (for openai>=1.0.0)**
client = openai.OpenAI(api_key=OPENAI_API_KEY)


class State(TypedDict):
    """Graph state containing research topic, generated content, and human feedback."""
    research_topic: str
    generated_text: Annotated[list[str], add_messages]
    human_feedback: Annotated[list[str], add_messages]


def first_node(state: State):
    """Uses GPT-4o to generate content based on a research topic with human feedback incorporated."""
    print("\n[first_node] Generating content using GPT-4o...")
    topic = state["research_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No feedback yet"]

    # **Define GPT-4o Prompt**
    prompt = f"""
    Research Topic: {topic}
    Human Feedback: {feedback[-1] if feedback else 'No feedback yet'}

    Generate a structured and well-written piece on the given research topic.
    Consider previous human feedback to refine the response. 
    """

    # **Call GPT-4o API (Fixed for openai>=1.0.0)**
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert research assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = response.choices[0].message.content
    print(f"[first_node] Generated text:\n{generated_text}\n")

    return {"generated_text": [generated_text], "human_feedback": feedback}


def human_node(state: State):
    """Human intervention node - loops back to first_node unless input is 'done'."""
    print("\n[human_node] Awaiting human feedback...")

    generated_text = state["generated_text"]

    # Interrupt to get user feedback
    user_feedback = interrupt(
        {"generated_text": generated_text, "message": "Provide feedback or type 'done' to finish."})
    print(f"[human_node] Received human feedback: {user_feedback}")

    # If user types 'done', transition to end_node
    if user_feedback.lower() == "done":
        return Command(update={"human_feedback": state["human_feedback"] + ["Finalized"]}, goto="end_node")

    # Otherwise, update feedback and return to first_node for re-generation
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="first_node")


def end_node(state: State):
    """Final node"""
    print("\n[end_node] Process finished.")
    print("Final Generated Text:", state["generated_text"][-1])
    print("Final Human Feedback:", state["human_feedback"])
    return {"generated_text": state["generated_text"], "human_feedback": state["human_feedback"]}


# **Building the Graph**
graph_builder = StateGraph(State)
graph_builder.add_node("first_node", first_node)
graph_builder.add_node("human_node", human_node)
graph_builder.add_node("end_node", end_node)

# **Define the Flow**
graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("first_node", "human_node")
graph_builder.add_edge("human_node", "first_node")  # Loop back unless 'done'
graph_builder.add_edge("human_node", "end_node")  # Exit if 'done'

graph_builder.set_finish_point("end_node")

# **Enable Interrupt Mechanism**
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# **Thread Configuration**
thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

# **Start the Graph Execution**
research_topic = input("Enter your research topic: ")
initial_state = {"research_topic": research_topic, "generated_text": [], "human_feedback": []}

for chunk in graph.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        # print("-----")
        # print(f"[Node]: {node_id}")
        # print(f"[Output]: {value}")

        # If we reach an interrupt, continuously ask for human feedback
        if node_id == "__interrupt__":
            while True:
                user_feedback = input("Provide feedback (or type 'done' to finish): ")

                # Resume the graph execution with the user's feedback
                graph.invoke(Command(resume=user_feedback), config=thread_config)

                # Exit loop if user says "done"
                if user_feedback.lower() == "done":
                    break
