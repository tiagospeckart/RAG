from langchain.agents import AgentType, initialize_agent

def create_agent_executor(llm, tools):
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
