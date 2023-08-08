from flask import Flask, jsonify, request, json
from ai_chat import ai_chat
from ai_assist import ai_assist
from ai_assist_tools import ai_assist_tools
from config.config import APIKEY, CHAT_MODEL, ASSIST_MODEL

from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ChatMessageHistory

app = Flask(__name__)

def isAuthorized(headers):
    auth = headers.get("X-APIKEY")
    if auth == APIKEY:
        return True
    else:
        return False



@app.route('/status', methods=['GET'])
async def status_request():
    return jsonify({"message": "Alive"}), 200
    

@app.route('/chat', methods=['POST'])
async def ai_chat_func():
    if isAuthorized(request.headers) == True:

        print("request: ")
        print(request.get_json())
        rawData = request.get_json()
        aiResponse = await ai_chat(rawData, chatAi)
        response = app.response_class(
            response=aiResponse,
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


@app.route('/chat_reinit_mem', methods=['POST'])
async def ai_chat_reinit_mem_func():
    if isAuthorized(request.headers) == True:

        rawData = request.get_json()
        print("rawData", rawData)
        chatHistory = rawData["conversationData"]["chatHistory"]

        chatAi = await initializeChatAiWithExistingMemories(chatHistory)

        response = app.response_class(
            response="done",
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401


@app.route('/chat_reinit', methods=['POST'])
async def ai_chat_reinit_func():
    if isAuthorized(request.headers) == True:
        chatAi = initializeChatAi()
        response = app.response_class(
            response="done",
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401




@app.route('/assist', methods=['POST'])
async def ai_assist_func():
    if isAuthorized(request.headers) == True:

        print("request: ")
        print(request.get_json())
        rawData = request.get_json()
        aiResponse = await ai_assist(rawData, assistantAi)
        response = app.response_class(
            response=aiResponse,
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    
@app.route('/assist_tools', methods=['POST'])
async def ai_assist_tools_func():
    if isAuthorized(request.headers) == True:

        print("request: ")
        print(request.get_json())
        rawData = request.get_json()
        aiResponse = await ai_assist_tools(rawData, assistantAiWithTools)
        response = app.response_class(
            response=aiResponse,
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    


def initializeAssistantAi():
    assist_llm_path = (ASSIST_MODEL)
    sum_llm_path =  ("cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q2_K.bin")
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    chat_llm = GPT4All(model=assist_llm_path, backend="gptj", callbacks=callbacks, verbose=True, max_tokens=2048, temp=1, top_p=0.4, top_k=40, n_batch=512, repeat_penalty=1.2, repeat_last_n=128)

    # Memory for summarizing all in conversation
    memory = ConversationBufferMemory(memory_key="chat_history_lines", input_key="input", return_messages=True)

    return {"callbacks": callbacks, "chat_llm": chat_llm, "memory": memory}

def initializeAssistantAiWithTools():
    assist_llm_path = (ASSIST_MODEL)
    sum_llm_path =  ("cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q2_K.bin")
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    chat_llm = GPT4All(model=assist_llm_path, backend="gptj", callbacks=callbacks, verbose=True, max_tokens=2048, temp=1, top_p=0.4, top_k=40, n_batch=512, repeat_penalty=1.2, repeat_last_n=128)

    # Memory for summarizing all in conversation
    memory = ConversationBufferMemory(memory_key="chat_history")

    return {"callbacks": callbacks, "chat_llm": chat_llm, "memory": memory}

def initializeChatAi():
    print("\n\nstarting to initialize model")
    chat_llm_path = (CHAT_MODEL)
    sum_llm_path =  ("cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q2_K.bin")
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    chat_llm = GPT4All(model=chat_llm_path, backend="gptj", callbacks=callbacks, verbose=True, max_tokens=2048, temp=1, top_p=0.4, top_k=40, n_batch=512, repeat_penalty=1.2, repeat_last_n=128)

    # Memory for summarizing all in conversation
    memory = ConversationSummaryMemory(llm=chat_llm, input_key="input")

    return {"callbacks": callbacks, "chat_llm": chat_llm, "memory": memory}

async def initializeChatAiWithExistingMemories(rawConversationData):
    print("\n\nstarting to initialize model with existing memory")
    chat_llm_path = (CHAT_MODEL)
    sum_llm_path =  ("cashman/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q2_K.bin")
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    chat_llm = GPT4All(model=chat_llm_path, backend="gptj", callbacks=callbacks, verbose=True, max_tokens=2048, temp=1, top_p=0.4, top_k=40, n_batch=512, repeat_penalty=1.2, repeat_last_n=128)
    
    
    # Memory for summarizing all in conversation    
    history = ChatMessageHistory()
    if( len(rawConversationData) > 2):
        print("\n\n")
        for idx, x in enumerate(rawConversationData):
            if idx % 2 == 0 and idx+1 < len(rawConversationData):
                if idx >= len(rawConversationData) - 20:
                    if idx % 2 == 0:
                        print("input:", x)
                        history.add_user_message(x)
                    else:
                        print("output: ", x)
                        history.add_ai_message(x)
        print("\n\n")

    memory = ConversationSummaryMemory.from_messages(
        llm=chat_llm,
        chat_memory=history,
        return_messages=True,
        input_key="input"
    )
    memory.buffer
    return {"callbacks": callbacks, "chat_llm": chat_llm, "memory": memory}

chatAi = initializeChatAi()
assistantAi = initializeAssistantAi()
assistantAiWithTools = initializeAssistantAiWithTools()