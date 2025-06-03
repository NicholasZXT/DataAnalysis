import os
import json
import requests

# --- Ollama 本地部署 ---
API_KEY = 'Empty'
LLM_URL = 'http://localhost:11434'
# MODEL = 'qwen2.5:7b'
MODEL = 'qwen3:8b'
# MODEL = 'qwen2.5:14b'
# MODEL = 'qwen3:14b'


def llm_chat(sys_msg, user_msg, stream=False, max_tokens=512, top_p=0.9, temperature=0.9, stop='<im_end>'):
    prompt_msgs = [
        {'role': 'system', 'content': sys_msg},
        {"role": "user", "content": user_msg}
    ]
    headers = {'Content-Type': 'application/json'}
    body = {
        # 'history': history,
        'model': MODEL,
        'messages': prompt_msgs,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'temperature': temperature,
        'repetition_penalty': 1.05,
        'stop': stop,
        'stream': stream
    }
    if not stream:
        try:
            res = requests.post(url=LLM_URL, headers=headers, json=body, timeout=600)
            res_json = res.json()
        except Exception as e:
            print(e)
            res_json = None
        return res_json
    if stream:
        try:
            res_stream = requests.post(url=LLM_URL, headers=headers, json=body, timeout=600)
            for chunk in res_stream.iter_lines():
                print(chunk.decode('utf-8'))
        except Exception as e:
            print(e)


def test_llm_chat():
    print("--> test_llm_chat running")
    msgs = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {"role": "user", "content": "RTX 4060Ti-16GB跑本地大模型怎么样？/no_think"}
    ]
    headers = {'Content-Type': 'application/json'}
    body = {
        # 'history': history,
        'model': MODEL,
        'messages': msgs,
        'stream': False,
        'think':  False,
        # 'stream': True,
        # 'think':  True,
    }
    res = requests.post(url=LLM_URL + '/api/chat', headers=headers, json=body, timeout=600)
    # print(res)
    res_json = res.json()
    print(res_json)
    # res_stream = requests.post(url=LLM_URL + '/api/chat', headers=headers, json=body, timeout=600, stream=True)
    # print(res_stream.status_code)
    # for chunk in res_stream.iter_lines():
    #     print(chunk.decode('utf-8'), end='')


def main():
    print("main running...")
    test_llm_chat()
    print("main end.")


if __name__ == '__main__':
    main()
