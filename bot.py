import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained(r'D:\.vscode\Cril\Git\AI\models\trained_model')
model = GPT2LMHeadModel.from_pretrained(r'D:\.vscode\Cril\Git\AI\models\trained_model')

# Функция генерации ответа
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Команда /start
def start(update: Update, _: CallbackContext) -> None:
    update.message.reply_text('Привет! Я чат-бот GyperBrain. Задай мне вопрос!')

# Обработка сообщений
def handle_message(update: Update, _: CallbackContext) -> None:
    user_message = update.message.text
    response = generate_response(user_message)
    update.message.reply_text(response)

# Обработка ошибок
def error(update: Update, context: CallbackContext) -> None:
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main() -> None:
    # Вставьте сюда токен вашего бота
    token = '7296874271:AAG8R3sksR-LfTSu0jtFO_VsEF5p7ewVSMk'

    # Создание Updater и передача ему токена вашего бота
    updater = Updater(token)

    # Получение диспетчера для регистрации обработчиков
    dispatcher = updater.dispatcher

    # Регистрация обработчиков команд и сообщений
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(filters.text & ~filters.command, handle_message))

    # Регистрация обработчика ошибок
    dispatcher.add_error_handler(error)

    # Запуск бота
    updater.start_polling()

    # Ожидание завершения работы
    updater.idle()

if __name__ == '__main__':
    main()
