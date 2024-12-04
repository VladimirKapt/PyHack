import telebot
from add_document import add_document
import llama_model 
from qdrant_client import QdrantClient
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, BartForConditionalGeneration, BartTokenizer
import torch
import bart_model
import gpt2_model
import my_token

bot = telebot.TeleBot(my_token.BOT_TOKEN)

qdrant_client = QdrantClient(host="localhost", port=6333)
# Загрузка модели и токенизатора для DPRQuestionEncoder
tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, 'Привет! Пришли мне файлы')

# Обработчик файлов
@bot.message_handler(content_types=['document'])
def handle_document(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open('received_file', 'wb') as new_file:
        new_file.write(downloaded_file)

    add_document(bot, message)

@bot.message_handler(commands=['ask'])
def ask(message):   

    new_m = bot.send_message(message.chat.id, 'Собираю данные')

    query = message.text.replace('/ask', '').strip()  

    # Токенизируем запрос и генерируем вектор
    query_vector = generate_vectors(query)

    relevant_data = retrieve_relevant_data_from_qdrant(query_vector)


    prompt = llama_model.create_prompt(query, relevant_data)

    response = llama_model.generate_response(prompt)

    bot.delete_message(message.chat.id, new_m.message_id)
    bot.send_message(message.chat.id, response)

def retrieve_relevant_data_from_qdrant(query):
    
    search_result = qdrant_client.search(
        collection_name="my_collection",
        query_vector=query,  
        limit=5  # Количество релевантных результатов
    )

    relevant_data = [hit.payload for hit in search_result]

    return relevant_data

def generate_vectors(text):
        inputs = tokenizer_dpr(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            vector = model_dpr(**inputs).pooler_output.squeeze().numpy()
        return vector.tolist()

# Запуск бота
bot.polling(none_stop=True) 