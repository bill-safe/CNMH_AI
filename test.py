import requests
import json

def call_ollama_api(input_text):
    url = "http://localhost:11434/api/chat"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "model": "your_model_name_here",  # 替换为你的模型名称
        "messages": [{"role":"user", "content": input_text}],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code} - {response.text}"

# 使用方法
input_text = "Your input text here"
response = call_ollama_api(input_text)
print(response)
