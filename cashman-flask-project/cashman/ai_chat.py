from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
import json
from flask import jsonify






async def ai_chat(rawData, llmData):
    conversationData = rawData["conversationData"]
    rawConversationData = conversationData["chatHistory"]
    ai_persona = conversationData["aiPersona"]
    ai_name = conversationData["aiName"]

    callbacks = llmData["callbacks"]
    chat_llm = llmData["chat_llm"]

    memory = llmData["memory"]

    
    template = """
    %s
    Summary of conversation:
    {history}
    Human: {input}
    %s:""" % (ai_persona, ai_name)
    prompt = PromptTemplate(template=template, input_variables=["history", "input"])


    # Verbose is required to pass to the callback manager
    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends

    llm_chain = LLMChain(prompt=prompt, llm=chat_llm, memory=memory)

    returnVal = llm_chain.predict(history=conversationData["conversationSummary"], input=conversationData["input"])
    
    print("input: ", conversationData["input"])
    print("output: ", returnVal)
    memory.save_context({"input": conversationData["input"]}, {"output": returnVal})
    data = {}
    data['answer'] = returnVal
    data['conversationSummary'] = memory.buffer
    json_data = json.dumps(data)
    return json_data
