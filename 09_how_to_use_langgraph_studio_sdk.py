import asyncio
import os

from dotenv import load_dotenv
from langgraph_sdk import get_client

load_dotenv()
LANGSMITH_API_KEY = os.environ.get('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# Replace this with the URL of your own deployed graph
URL = "https://rajib-langgraph-studio04-5ea7b9b3fcda5429a844191acfd6d8a2.default.us.langgraph.app"
client = get_client(url=URL)

input = {"messages": [{"role": "user", "content": "Why is Eiffel tower famous?"}]}


async def search_assistant():
    assistants = await client.assistants.search()
    print(assistants)
    assistant = assistants[0]
    thread = await client.threads.create()
    return assistant, thread


async def run_assistant():
    assistant, thread = await search_assistant()
    chunks = client.runs.stream(
        thread['thread_id'],
        assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    )
    async for chunk in chunks:
        print(chunk)


asyncio.run(run_assistant())

#
# # Search all hosted graphs
# assistants = client.assistants.search()
# # In this example we select the first assistant since we are only hosting a single graph
# assistant = assistants[0]
#
# print(assistant)
#
# # We create a thread for tracking the state of our run
# thread = client.threads.create()
# # thread = await client.threads.create()
# input = {"messages":[{"role": "user", "content": "where is TajMahal?"}]}
#
# chunks = client.runs.stream(
#         thread['thread_id'],
#         assistant["assistant_id"],
#         input=input,
#         stream_mode="updates",
#     )
#
# for chunk in chunks:
#     if chunk.data and chunk.event != "metadata":
#         print(chunk.data)
