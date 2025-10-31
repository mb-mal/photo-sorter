# PhotoSorter

AI-ассистированная система классификации и сортировки фотографий с использованием локальной модели Ollama и векторной базы данных Qdrant. Система автоматически определяет категории и подкатегории, создавая динамическую таксономию на основе содержимого изображений.

> 🐳 **Быстрый старт с Docker**: `docker-compose up --build` (см. [раздел Docker](#-docker-быстрый-старт))

## 🎯 Возможности

- **Интеллектуальная классификация**: Многоэтапный AI-пайплайн для точной категоризации
- **Динамические категории**: Автоматическое создание подкатегорий на основе содержимого
- **Векторный поиск**: Использование CLIP-эмбеддингов для поиска похожих фотографий
- **EXIF метаданные**: Учет метаданных камеры и времени съемки
- **99.99% точность**: Постоянное улучшение через контекст похожих фото

## 📋 Требования

- Python 3.13+
- Ollama с установленной моделью
- Qdrant (локальный сервер)
- ~40GB свободного места для моделей

## 🚀 Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/mb-mal/photo-sorter.git
cd photo-sorter
```

### 2. Создание виртуального окружения Python

```bash
python3 -m venv .
source bin/activate  # Linux/macOS
# или
.\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Установка и запуск Qdrant

**Docker (рекомендуется):**

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Или локальная установка:**

```bash
# macOS
brew install qdrant

# Linux
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant
```

Qdrant будет доступен по адресу: `http://localhost:6333`

### 5. Установка и настройка Ollama

**Установка Ollama:**

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Скачайте установщик с https://ollama.ai/download
```

**Загрузка модели:**

```bash
ollama pull gemma3:27b-it-qat
```

Рекомендуемые модели:

- `gemma3:27b-it-qat` - быстрая и точная (по умолчанию)
- `qwen3-vl:32b` - максимальная точность

### 6. Подготовка директорий

```bash
mkdir img  # Исходные фотографии
mkdir sorted_photos  # Отсортированные фотографии (создастся автоматически)
```

Поместите фотографии для обработки в директорию `img/`.

## 🐳 Docker (быстрый старт)

Для удобного запуска всего стека используйте Docker Compose:

### Быстрый старт с Docker

```bash
# Клонируйте репозиторий
git clone https://github.com/mb-mal/photo-sorter.git
cd photo-sorter

# Подготовьте директории
mkdir -p img sorted_photos data

# Поместите фотографии в img/

# Запустите Qdrant и приложение
docker-compose up --build
```

### Запуск только Qdrant в Docker

Если Ollama установлен локально, можно запустить только Qdrant:

```bash
docker-compose up qdrant -d
```

### Запуск приложения в Docker с локальной Ollama

Для работы с локально установленной Ollama используйте переменную окружения:

```bash
# macOS/Windows
docker-compose run --rm photosorter

# Linux (если Ollama на хосте)
OLLAMA_HOST=host.docker.internal:11434 docker-compose run --rm photosorter
```

### Параметры запуска в Docker

```bash
# Ограниченное количество фото для тестирования
docker-compose run --rm photosorter --limit 10

# Тестовый режим
docker-compose run --rm photosorter --dry-run

# С указанием модели
docker-compose run --rm photosorter --model gemma3:27b-it-qat --verbose
```

### Структура Docker-образов

- **`qdrant`**: Векторная база данных (порт 6333)
- **`photosorter`**: Основное приложение (запускается по требованию)

### Volumes

Docker Compose автоматически создает volumes для:
- `img/` - исходные фотографии (read-only)
- `sorted_photos/` - отсортированные фотографии
- `data/` - SQLite база данных
- HuggingFace cache - для кэширования моделей между запусками

### Важные заметки для Docker

1. **Ollama должен быть запущен локально** (вне Docker) или использовать отдельный контейнер
2. **Первый запуск** может занять время для загрузки CLIP модели (~1.5GB)
3. Для **Linux** убедитесь, что Ollama доступен через `host.docker.internal` или используйте сетевой режим `host`

## 💻 Использование (локальная установка)

### Базовый запуск

```bash
source bin/activate
python photo_sorter.py
```

### С указанием модели

```bash
python photo_sorter.py --model gemma3:27b-it-qat
```

### Обработка ограниченного количества фото (для тестирования)

```bash
python photo_sorter.py --limit 10
```

### Тестовый режим (без перемещения файлов)

```bash
python photo_sorter.py --dry-run
```

### Принудительная переобработка

```bash
python photo_sorter.py --force
```

### Указание кастомных директорий

```bash
python photo_sorter.py --source /path/to/photos --destination /path/to/output
```

## 📁 Структура директорий

- **`img/`** — исходные фотографии для обработки
- **`sorted_photos/`** — отсортированные фотографии по категориям:
  - `business/` — рабочая категория
    - `diagrams/` — графики и схемы
    - `events/` — события и конференции
    - `receipts/` — чеки и билеты
    - `travel/` — деловые поездки
    - `none/` — прочие бизнес-фото
  - `personal/` — личная категория
    - `pets/` — домашние животные
    - `school/` — учеба
    - `misc/` — прочее
  - `sightseeing/` — туризм и достопримечательности
    - `architecture/` — архитектура
    - `landmarks/` — известные места
    - `cityscape/` — городские пейзажи

## 🛠️ Параметры командной строки

| Параметр  | Описание                                                                           | По умолчанию       |
| ----------------- | ------------------------------------------------------------------------------------------ | ----------------------------- |
| `--source`      | Директория с исходными фото                                        | `img`                       |
| `--destination` | Директория для отсортированных фото                        | `sorted_photos`             |
| `--model`       | Модель Ollama                                                                        | `gemma3:27b-it-qat`         |
| `--temperature` | Температура сэмплирования (0.0 = детерминированно) | `0.0`                       |
| `--limit`       | Максимальное количество фото для обработки           | без ограничений |
| `--force`       | Переобработка уже обработанных фото                        | `false`                     |
| `--dry-run`     | Тестовый режим (без перемещения файлов)                   | `false`                     |
| `--verbose`     | Подробное логирование                                                  | `false`                     |

## 🔄 Сброс состояния

Для очистки базы данных и возврата фото в исходную директорию:

```bash
./reset.sh
```

Этот скрипт:

- Переместит все фото из `sorted_photos/` обратно в `img/`
- Очистит SQLite базу данных
- Удалит коллекции в Qdrant

## 📊 Архитектура

Система использует многоэтапный процесс классификации:

1. **STEP 0**: Генерация детального описания изображения через Ollama
2. **STEP 1**: Определение топ-уровневой категории (business/personal/sightseeing)
3. **STEP 2**: Векторный поиск похожих фото в Qdrant
4. **STEP 3**: Определение/создание подкатегории с учетом контекста

### Компоненты

- **EmbeddingService**: Генерация CLIP-эмбеддингов (768-dim, ViT-L-14)
- **VectorIndex**: Взаимодействие с Qdrant для хранения и поиска
- **IntelligentClassifier**: Оркестрация многоэтапной классификации
- **PhotoRepository**: Управление SQLite базой данных

## 🧪 Тестирование

```bash
pytest tests/
```

## 📝 Логирование

Все операции логируются с цветной индикацией:

- 🔵 Синий — информационные сообщения
- 🟢 Зеленый — успешные операции
- 🟡 Желтый — предупреждения
- 🔴 Красный — ошибки

## 📄 Лицензия

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) file for details.
