import os

from dotenv import load_dotenv
from langgraph.func import task, entrypoint
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# Initialize OpenAI client
client = OpenAI()

@task
def generate_punchline(topic: str):
    """Generates a punchline for the given topic."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Generate a powerful punchline for the topic: {topic}"}]
    )
    return response.choices[0].message.content


@task
def generate_blog(topic: str):
    """Generates an impactful blog on the given topic."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Write an insightful and impactful blog on the topic: {topic}"}]
    )
    return response.choices[0].message.content


@task
def format_medium_article(topic, punchline, blog):
    """Formats the punchline and blog into a medium-style article."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Format the following into a Medium-style article:\n\n"
                                              f"**Topic:** {topic}\n\n"
                                              f"**Punchline:** {punchline}\n\n"
                                              f"**Blog:**\n{blog}"}]
    )
    return response.choices[0].message.content


# Define the workflow
@entrypoint()
def medium_workflow(topic: str):
    punchline_fut = generate_punchline(topic)
    blog_fut = generate_blog(topic)
    medium_article_fut = format_medium_article(
        topic, punchline_fut.result(), blog_fut.result()
    )
    return medium_article_fut.result()


# Invoke the workflow
for step in medium_workflow.stream("The Future of AI", stream_mode="updates"):
    for node_id,result in step.items():
        print(node_id)
        print(result)
