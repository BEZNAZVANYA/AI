import json
import pandas as pd
from datasets import Dataset

# Загрузка и подготовка собственных данных
with open('D:\.vscode\Cril\Git\AI\data\custom_dataset.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_pandas(pd.DataFrame(data))

# Сохранение подготовленных данных
dataset.save_to_disk('D:\.vscode\Cril\Git\AI\data\prepared_dataset')
