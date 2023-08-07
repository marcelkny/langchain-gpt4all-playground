from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
import json
from flask import jsonify

async def ai_assist(rawData, assistData):
    conversationData = rawData["conversationData"]
    rawConversationData = conversationData["chatHistory"]
    ai_persona = "You are a helpful AI assistant. You always try to provide a direct answer. If you dont know the answer, you say that you don't know it"
    ai_name = conversationData["aiName"]

    callbacks = assistData["callbacks"]
    chat_llm = assistData["chat_llm"]

    memory = assistData["memory"]
    
    template = """
    %s
    Current conversation:
    {chat_history_lines}
    Human: {input}
    %s:""" % (ai_persona, ai_name)
    prompt = PromptTemplate(template=template, input_variables=["input", "chat_history_lines"])



    if( len(rawConversationData) > 2):
        print("\n")
        for idx, x in enumerate(rawConversationData):
            if idx % 2 == 0 and idx+1 < len(rawConversationData):
                if idx >= len(rawConversationData) - 6:
                    print("saving to conversation:")
                    memory.save_context({"input": x}, {"output": rawConversationData[idx+1]})
            print(idx, x)
        print("\n")


    # Verbose is required to pass to the callback manager
    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends

    llm_chain = LLMChain(prompt=prompt, llm=chat_llm, memory=memory)

    returnVal = llm_chain.predict(input=conversationData["input"], chat_history_lines=conversationData["chatHistory"])
    
    
    print("\ninput: ", conversationData["input"])
    print("\noutput: ", returnVal)


    testArr = []
    conv_mem_arr = memory.load_memory_variables({})['chat_history_lines']
    for x in conv_mem_arr:
        stripped1 = str(x).replace(' additional_kwargs={} example=False',"")
        stripped2 = stripped1.replace("content=' ","'")
        finalstripped = stripped2.replace("content='","'")
        testArr.append(finalstripped)
    data = {}
    data['answer'] = returnVal
    data['currentConversation'] = testArr
    json_data = json.dumps(data)
    return json_data
