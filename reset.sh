#!/bin/bash
# PhotoSorter Reset Script
# Возвращает все фото в img/ и очищает базы данных

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Сброс PhotoSorter к исходному состоянию..."
echo ""

# 1. Возврат фото
echo "📸 Возврат фото в img/..."
if [ -d "sorted_photos" ]; then
    find sorted_photos -type f -name "*.jpg" -exec mv {} img/ \; 2>/dev/null || true
    find sorted_photos -type d -empty -delete 2>/dev/null || true
    echo "   ✅ Фото возвращены"
else
    echo "   ⏩ sorted_photos/ не найден"
fi

# 2. Очистка SQLite
echo "🗄️  Очистка SQLite базы..."
if [ -f "photo_analysis.sqlite" ]; then
    sqlite3 photo_analysis.sqlite "DELETE FROM photo_analysis; VACUUM;" 2>/dev/null || true
    echo "   ✅ SQLite очищена"
else
    echo "   ⏩ photo_analysis.sqlite не найден"
fi

# 3. Очистка Qdrant
echo "🔍 Очистка Qdrant..."
if command -v python &> /dev/null; then
    source bin/activate 2>/dev/null || true
    python - <<'PYEOF' 2>/dev/null || echo "   ⚠️  Qdrant недоступен (это нормально если он не запущен)"
from qdrant_client import QdrantClient
try:
    client = QdrantClient(url="http://localhost:6333")
    collections = [c.name for c in client.get_collections().collections]
    for name in collections:
        client.delete_collection(name)
        print(f"   ✅ Удалена коллекция: {name}")
    if not collections:
        print("   ✅ Qdrant уже пуст")
except Exception as e:
    print(f"   ⚠️  Qdrant недоступен: {e}")
PYEOF
else
    echo "   ⏩ Python не найден, пропуск Qdrant"
fi

# 4. Подсчёт фото
echo ""
PHOTO_COUNT=$(ls img/*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Готово! В img/ находится $PHOTO_COUNT фото"
echo ""
echo "Можно запускать:"
echo "  python photo_sorter.py --model gemma3:27b-it-qat"

