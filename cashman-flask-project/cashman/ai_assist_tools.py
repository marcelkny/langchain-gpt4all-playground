from langchain import PromptTemplate, LLMChain
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import AgentType, initialize_agent

import json
from flask import jsonify

async def ai_assist_tools(rawData, assistData):
    conversationData = rawData["conversationData"]
    ai_persona = "You are a helpful AI assistant. You always try to provide a direct answer. If you dont know the answer, you say that you don't know it"
    ai_name = conversationData["aiName"]

    callbacks = assistData["callbacks"]
    chat_llm = assistData["chat_llm"]

    memory = assistData["memory"]
    tools = []
    # WIKIPEDIA TOOL
    wikipedia = WikipediaAPIWrapper
    wikipedia_tool = Tool(
        name= "wikipedia",
        func= wikipedia.run,
        description= "useful for when you need to lookup a topic, country or person on wikipedia"
    )
    # DUCKDUCKGO TOOL
    search = DuckDuckGoSearchRun
    search_tool = Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="useful for when you need to do a search on the internet"
    )

    tools.append(wikipedia_tool)
    tools.append(search_tool)


    agent_chain = initialize_agent(tools, chat_llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    answer = agent_chain.run(input=conversationData["input"])
    print(answer)



    data = {}
    data['answer'] = "returnString"
    data['currentConversation'] = []
    json_data = json.dumps(data)
    return json_data
