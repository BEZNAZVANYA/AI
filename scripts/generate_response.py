import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загрузка обученной модели
tokenizer = GPT2Tokenizer.from_pretrained('D:\.vscode\Cril\Git\AI\models\trained_model')
model = GPT2LMHeadModel.from_pretrained('D:\.vscode\Cril\Git\AI\models\trained_model')

# Функция генерации ответа
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Пример использования
prompt = "Привет, как тебя зовут?"
response = generate_response(prompt)
print(response)
