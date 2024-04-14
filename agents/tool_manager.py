import requests

from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool

def comments_func(__query__):
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/comments/1")
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error fetching comments: {e}"


def posts_func(__query__):
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error fetching posts: {e}"
    

def todos_func(__query__):
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error fetching todos: {e}"
    

def create_verctor_db_tool(retriever):
    return create_retriever_tool(
        retriever,
        "t-store-retriever",
        "Vector database retriever to get information about T-Store",
    )
    

def get_tools(retriever):
    return [
        Tool(
            name="comments-api",
            func=comments_func,
            description="Use this tool to retrieve 'comments' from 'jsonplaceholder.typicode.com'"
        ),
        Tool(
            name="posts-api",
            func=posts_func,
            description="Use this tool to retrieve 'posts' from 'jsonplaceholder.typicode.com'"
        ),
        Tool(
            name="todos-api",
            func=todos_func,
            description="Use this tool to retrieve 'todos' from 'jsonplaceholder.typicode.com'"
        ),
        create_verctor_db_tool(retriever)
    ]
