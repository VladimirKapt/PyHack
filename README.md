# PyHack

Проект позволяет загружать слюбые виды документов(.pdf,.word,.csv) и с помощию LLM получать ответ на любой интересующий вопрос

Пример работы:

![image](https://github.com/user-attachments/assets/cd945cfe-b5f4-4143-b841-bf97c3f2a7e1)

Для хранения виспользуеться векторная бд Qdrant развернутая в докер контейнере. Загружаемые документы переводяться в векторную форму и загружаються внутрь бд.
При запросе пользователя вопрос переводиться в векторную форму и производиться поиск n более похожих векторов в бд, учитывая их содержание LLM отвечает на поставленный вопрос.

Для запуска необходимо:
  1. Поставить Docker desktop и запустить его.
  2. Загрузить (pull) образа Docker:  docker pull qdrant/qdrant
  3. Запустить Qdrant с помощью Docker: docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
  4. Установить пакеты (при необходимости): pip install pyTelegramBotAPI qdrant-client transformers torch requests
  5. Изменить токен бота в файле main.py
  6. Скачать Ollama
  7. Запустить модель: ollama run llama3.2
  8. Отправитьь документы боту
  9. Спросить через команду /ask

P.S. не конечная версия, возможны доработки для улучшения юзер стори.
