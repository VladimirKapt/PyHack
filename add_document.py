from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize
from TextParser import TextParser

nltk.download('punkt')
nltk.download('punkt_tab')

collection_name = "my_collection"

def add_document(bot, message):
    print("НАЧАЛО ОБРАБОТКИ")
    try:
        bot.send_message(message.chat.id, 'Загружаем файл в бд')

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open('received_file', 'wb') as new_file:
            new_file.write(downloaded_file)

        file_type = message.document.mime_type
        print(file_type)

        parser = TextParser()
        text = None
        if file_type == "text/plain":
            text = parser.parse_txt('received_file')
        elif file_type == "application/pdf":
            text = parser.parse_pdf('received_file')
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = parser.parse_word('received_file')
        elif file_type == "text/csv":
            text = parser.parse_csv('received_file')

        print(text)

        # Инициализация Qdrant клиента
        qdrant_client = QdrantClient(host="localhost", port=6333)

        # Проверка существования коллекции
        collections = qdrant_client.get_collections()
        if collection_name not in [collection.name for collection in collections.collections]:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print(f"Коллекция {collection_name} создана.")
        else:
            print(f"Коллекция {collection_name} уже существует.")

        # Загрузка модели и токенизатора для DPRQuestionEncoder
        tokenizer_dpr = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model_dpr = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        def generate_vectors(text):
            inputs = tokenizer_dpr(text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                vector = model_dpr(**inputs).pooler_output.squeeze().numpy()
            return vector.tolist()

        # для загрузки текста в Qdrant
        def upload_text_to_qdrant(collection_name, text):
            sentences = text.split('\n\n')

            vectors = [generate_vectors(sentence) for sentence in sentences]

            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    {"id": i, "vector": vector, "payload": {"text": sentence}}
                    for i, (sentence, vector) in enumerate(zip(sentences, vectors))
                ]
            )
            print("Текст успешно загружен в Qdrant.")

        upload_text_to_qdrant(collection_name, text)

        print("Векторы успешно загружены в Qdrant.")

        bot.send_message(message.chat.id, 'Файл успешно обработан.')
    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {e}')