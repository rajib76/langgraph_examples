import operator
import os
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI()
client = OpenAI()


# The shared state of the agents
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

summary_system_prompt = """
Extract the key discussion items and action items from the provided chat transcript between a bank customer 
and a laon agent. After extracting the relevant content, 
please draft an email that can be sent to the traveller. Please only share the draft email, DO NOT ADD anything else.
"""

user_prompt = """
Below is the chat transcript between a bank customer and a loan agent
chat_transcript:
Loan Agent (LA): Hello! I'm Alex, your loan agent. How can I assist you today?
Customer (C): Hi Alex, I’m interested in refinancing my house and wanted to explore the options available.
LA: Great! To get started, could you please provide me with some basic information about your current mortgage? Like your current interest rate and the remaining loan balance?
C: Sure, my current mortgage has a balance of $250,000 with an interest rate of 4.5%.
LA: Thank you for the information. How long have you been paying off this mortgage?
C: It's been around 10 years now.
LA: Perfect. Given the details, we can look into several refinancing options. Are you aiming to lower your monthly payments, or are you more interested in changing the term of your loan?
C: I’d like to lower my monthly payments if possible.
LA: Understood. One option could be switching to a loan with a lower interest rate. Given current rates, you might qualify for something significantly lower than 4.5%. We also have options to extend the loan term if that interests you.
C: I think a lower rate sounds good. What would be the next steps to get an exact rate based on my financial situation?
LA: Next, we'll need to complete a formal application and run a credit check. You’ll also need to submit some documentation, like recent pay stubs, tax returns, and information on your assets and liabilities.
C: Can you send me a list of all the documents I need to gather?
LA: Absolutely, I’ll email you a detailed list right after our chat. Once we receive your documents, we can process your application and give you a few specific rate options.
C: That sounds good. How long does the process usually take?
LA: Typically, from application to closing, it can take about 30 to 45 days, depending on how quickly you can provide the necessary documentation.
C: Okay, I’ll gather the documents and send them over as soon as I can. Thanks for the info!
LA: You’re welcome! I'll send the email with the document list and follow up once we receive everything. If you have any questions in the meantime, feel free to contact me.
C: Will do. Thanks for your help, Alex!
LA: My pleasure! Looking forward to working with you. Have a great day!
"""

class MeetingSummary(BaseModel):
    discussion_point: str


class ActionItems(BaseModel):
    action_item: str


class Email(BaseModel):
    meeting_summary: list[MeetingSummary]
    customer_action_items: list[ActionItems]
    loan_agent_action_items: list[ActionItems]


def generate_summary(state):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": summary_system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=Email,
    )

    email = completion.choices[0].message.parsed
    return {"messages": [email]}

graph_builder = StateGraph(State)
graph_builder.add_node("generate_summary", generate_summary)
graph_builder.set_entry_point("generate_summary")
graph_builder.set_finish_point("generate_summary")
graph = graph_builder.compile()
for event in graph.stream({"messages": ["user_input"]}):
    for key, value in event.items():
        if key == "generate_summary":
            meeting_summaries = value["messages"][0].meeting_summary
            customer_action_items = value["messages"][0].customer_action_items
            loan_agent_action_itens = value["messages"][0].loan_agent_action_items
            print("#### Meeting Summary")
            for summary in meeting_summaries:
                print(summary.discussion_point)

            if len(customer_action_items)>0:
                print("#### Traveller Action Items")
                for action_item in customer_action_items:
                    print(action_item.action_item)

            if len(loan_agent_action_itens) > 0:
                print("#### Travel Agent Action Items")
                for action_item in loan_agent_action_itens:
                    print(action_item.action_item)