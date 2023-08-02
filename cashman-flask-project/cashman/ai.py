from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory

local_path = (
        "cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q3_K_L.bin"  # replace with your desired local file path
    )

async def ai_chat():
    
    
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True, max_tokens=512, temp=0.8, top_p=0.4, top_k=40, n_batch=128, repeat_penalty=1.18, repeat_last_n=64)
    
    # Memory for current conversation
    conv_memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input", return_messages=True)

    # Memory for summarizing all in conversation
    summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
    
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    
    ai_persona = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
    
    template = """ %s

    Summary of conversation:
    {history}
    Current conversation:
    {chat_history_lines}
    Human: {input}
    AI:""" % (ai_persona)
    prompt = PromptTemplate(template=template, input_variables=["history", "input", "chat_history_lines"])


    # Verbose is required to pass to the callback manager
    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends

    

    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

    returnVal = llm_chain.predict(history="", input="who are you?", chat_history_lines="")
    
    print("\nAI Response:")
    print(returnVal)
    print("...")
    print("Summary of all of Conversation:")
    print(summary_memory.buffer)
    print("...")
    print("Memory of current Conversation:")
    print(conv_memory.buffer)
    print("...")
    return {"answer": returnVal, "currentConversation": conv_memory.buffer, "conversationSummary": summary_memory.buffer}
