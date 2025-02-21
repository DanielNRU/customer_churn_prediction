FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем пакеты
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в контейнер
COPY . .

# Открываем порты для FastAPI и Gradio
EXPOSE 8000 7860

# Запуск основного скрипта
CMD ["python", "main.py"]
