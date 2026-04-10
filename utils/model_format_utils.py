
USER_TAG = "<<USER>>"
SYSTEM_TAG = "<<SYSTEM>>"
ASSISTANT_TAG = "<<ASSISTANT>>"


def format_system_prompt(system_prompt):
    return f"{SYSTEM_TAG}{system_prompt}"

def format_user_query(query):
    return f"{USER_TAG}{query}"

def format_assistant_response(response):
    return f"{ASSISTANT_TAG}{response}"


def query_to_message(encoded_lines):
    """
    client发到server的消息转为hf model apply_chat_template需要的格式
    使用如下自定义格式，通过换行维护对话历史，之后可参考其他库替换格式解析
    plain_text = "<<System>>You are a helpful assistant.\n
    <<User>> Tell me about large language models.\n
    <<Assistant>> Large language models are AI systems trained on vast amounts of text data to understand and generate human-like language."
    """
    msgs = []
    for line in encoded_lines:
        line = line.decode().strip()
        if line.startswith(USER_TAG):
            msgs.append({'role': 'user', 'content': line[len(USER_TAG):].strip()})
        elif line.startswith(ASSISTANT_TAG):
            msgs.append({'role': 'assistant', 'content': line[len(ASSISTANT_TAG):].strip()})
        elif line.startswith(SYSTEM_TAG):
            msgs.append({'role': 'system', 'content': line[len(SYSTEM_TAG):].strip()})
        else:
            msgs.append({'role': 'user', 'content': line})
    return msgs