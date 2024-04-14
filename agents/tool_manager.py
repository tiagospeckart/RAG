import requests

from langchain.agents import Tool

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
    

def get_tools():
    return [
        Tool(
            name="commentsapi",
            func=comments_func,
            description="Fetch comments from the API"
        ),
        Tool(
            name="postsapi",
            func=posts_func,
            description="Fetch posts from the API"
        ),
        Tool(
            name="todosapi",
            func=todos_func,
            description="Fetch todos from the API"
        )
    ]
