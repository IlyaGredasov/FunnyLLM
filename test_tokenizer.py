from dotenv import load_dotenv

from model.train import tokenize

load_dotenv()

s = "Это тестовый текст для проверки работы токенизатора"
print(tokenize(list(s.split())))
