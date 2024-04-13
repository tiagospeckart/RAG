import requests

from langchain.agents import Tool

def comments_func(__query__):
        if "comments" in __query__.lower():
            response = requests.get("https://jsonplaceholder.typicode.com/posts/1/comments")
            return response.json()
            # return f"This used the comments agent"
        else:
            return "No comments requested."


def get_app_version(__query__):
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
        response.raise_for_status()
        version = response.text
        return f"The current app version is {version}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching app version: {e}"
    

def get_tools():
    return [
        Tool(
            name="commentsapi",
            func=comments_func,
            description="Fetch comments from the API"
        ),
        Tool(
            name="GetUserId",
            func=get_app_version,
            description="Use this tool to get the current user id from the API"
        )
    ]
