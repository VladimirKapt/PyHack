import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from qdrant_client import QdrantClient
import requests
import json

# URL для обращения к API Ollama
ollama_url = "http://localhost:11434/api/generate"

ollama_model = "llama3.2"

qdrant_client = QdrantClient(host="localhost", port=6333)

tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

def create_prompt(query, relevant_data):
    # Создаем промпт, включающий запрос и релевантные данные
    prompt = f"Запрос: {query}\n\n"
    for idx, data in enumerate(relevant_data, 1):
        prompt += f"Ответь на поставленный вопрос используя эти релевантные данные {idx}: {data}\n"

    return prompt

def generate_vectors(text):
    # Токенизируем текст и генерируем вектор
    inputs = tokenizer_dpr(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        vector = model_dpr(**inputs).pooler_output.squeeze().numpy()
    return vector.tolist()

def retrieve_relevant_data_from_qdrant(query_vector):
    # Извлекаем релевантные данные из Qdrant
    search_result = qdrant_client.search(
        collection_name="my_collection",
        query_vector=query_vector,  
        limit=5  
    )

    relevant_data = [hit.payload for hit in search_result]

    return relevant_data

def generate_response(prompt):
    data = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(ollama_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        return f"Ошибка при отправке запроса: {e}"

    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("response", "Ошибка: ответ от сервера не содержит поле 'response'")
    else:
        return f"Ошибка: не удалось получить ответ от сервера (код {response.status_code})"

    