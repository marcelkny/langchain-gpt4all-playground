from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
import json
from flask import jsonify

chat_llm_path = ("cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q3_K_L.bin")
sum_llm_path =  ("cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q2_K.bin")


async def ai_chat(rawData):

    # Params for Model/Text-Generation
    model = rawData["modelParams"]

    

    
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    chat_llm = GPT4All(model=chat_llm_path, backend="gptj", callbacks=callbacks, verbose=True,
                  max_tokens=model["max_tokens"], temp=model["temp"], top_p=model["top_p"], top_k=model["top_k"],
                  n_batch=model["n_batch"], repeat_penalty=model["repeat_penalty"], repeat_last_n=model["repeat_last_n"])
    #sum_llm = GPT4All(model=sum_llm_path, backend="gptj", callbacks=callbacks, verbose=True,
    #              max_tokens=model["max_tokens"], temp=model["temp"], top_p=model["top_p"], top_k=model["top_k"],
    #              n_batch=model["n_batch"], repeat_penalty=model["repeat_penalty"], repeat_last_n=model["repeat_last_n"])
    
    # Memory for current conversation
    conv_memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input", return_messages=True)
    # Memory for summarizing all in conversation
    summary_memory = ConversationSummaryMemory(llm=chat_llm, input_key="input")
    memory = CombinedMemory(memories=[conv_memory, summary_memory])


    conversationData = rawData["conversationData"]
    rawConversationData = conversationData["chatHistory"]

    if( len(rawConversationData) > 2):
        print("\n")
        for idx, x in enumerate(rawConversationData):
            if idx % 2 == 0 and idx+1 < len(rawConversationData):
                if idx >= len(rawConversationData) - 10:
                    print("saving to conversation:")
                    conv_memory.save_context({"input": x}, {"output": rawConversationData[idx+1]})

                    if len(rawConversationData) % 5 == 0:
                        print("saving to summary:")
                        summary_memory.save_context({"input": x}, {"output": rawConversationData[idx+1]})
                
            print(idx, x)
        print("\n")

            
            

    ai_persona = conversationData["aiPersona"]
    ai_name = conversationData["aiName"]
    
    template = """
    %s
    Summary of conversation:
    {history}
    Current conversation:
    {chat_history_lines}
    Human: {input}
    %s:""" % (ai_persona, ai_name)
    prompt = PromptTemplate(template=template, input_variables=["history", "input", "chat_history_lines"])


    # Verbose is required to pass to the callback manager
    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends

    

    llm_chain = LLMChain(prompt=prompt, llm=chat_llm, memory=memory)

    returnVal = llm_chain.predict(history=conversationData["conversationSummary"], input=conversationData["input"], chat_history_lines=conversationData["chatHistory"])
    
    testArr = []
    conv_mem_arr = conv_memory.load_memory_variables({})['chat_history_lines']
    for x in conv_mem_arr:
        stripped1 = str(x).replace(' additional_kwargs={} example=False',"")
        stripped2 = stripped1.replace("content=' ","'")
        finalstripped = stripped2.replace("content='","'")
        testArr.append(finalstripped)
    data = {}
    data['answer'] = returnVal
    data['currentConversation'] = testArr
    data['conversationSummary'] = summary_memory.buffer
    json_data = json.dumps(data)
    return json_data
