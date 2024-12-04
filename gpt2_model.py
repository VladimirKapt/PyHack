from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import re

tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Загрузка модели и токенизатора gpt-2
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

def create_prompt(query, relevant_data):
    prompt = f"Запрос: {query}\n\n"
    for idx, data in enumerate(relevant_data, 1):
        prompt += f"Данные {idx}: {data['text']}\n"
    return prompt

def generate_vectors(text):
    inputs = tokenizer_dpr(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        vector = model_dpr(**inputs).pooler_output.squeeze().numpy()
    return vector.tolist()

def generate_response(prompt):
    inputs = tokenizer_gpt2(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model_gpt2.generate(
        inputs.input_ids,
        max_new_tokens=150, 
        num_return_sequences=1,
        temperature=0.7,  
        do_sample=True,  
        top_k=50,  
        top_p=0.95 
    )
    response = tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)
    return response

def clean_response(response):
    response = response.strip() 
    response = re.sub(r'\n+', '\n', response)  
    response = re.sub(r'\s+', ' ', response)  
    response = response[:500]
    return response

