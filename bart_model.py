from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer



# Загрузка модели и токенизатора для DPRQuestionEncoder
tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# Загрузка модели и токенизатора BART
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


def create_prompt(query, relevant_data):
    prompt = f"Запрос: {query}\n\n"
    for idx, data in enumerate(relevant_data, 1):
        prompt += f"Ответь на поставленный вопрос используя эти релевантные данные {idx}: {data}\n"

    return prompt

def generate_vectors(text):
    inputs = tokenizer_dpr(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        vector = model_dpr(**inputs).pooler_output.squeeze().numpy()
    return vector.tolist()

def generate_response(prompt):
    inputs = tokenizer_bart(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model_bart.generate(
        inputs.input_ids,
        max_length=150,  
        num_return_sequences=1,
        temperature=0.7,  
        do_sample=True,  
        top_k=50,  
        top_p=0.95  
    )
    response = tokenizer_bart.decode(outputs[0], skip_special_tokens=True)
    return response