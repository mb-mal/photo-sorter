#!/usr/bin/env python3
"""AI-assisted photo sorter leveraging a local Ollama deployment."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

from PIL import Image, ExifTags
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ollama import chat, ChatResponse


@dataclass(frozen=True)
class SorterConfig:
    source_dir: Path
    output_dir: Path
    database: Path
    model: str = "gemma3:27b-it-qat"
    temperature: float = 0.0
    retry_limit: int = 3
    retry_backoff_seconds: float = 3.0
    qdrant_collection: str = "photo_embeddings"
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/clip-ViT-L-14"  # 75.4% ImageNet (768-dim), лучше точность для классификации


class MetadataExtractor:
    """Extracts EXIF metadata and capture timestamps from images."""

    _EXIF_DATETIME_KEYS = (
        "DateTimeOriginal",
        "DateTimeDigitized",
        "DateTime",
    )

    _PROMPT_KEYS = {
        "Make",
        "Model",
        "LensModel",
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
    }

    def extract(self, image_path: Path) -> Dict[str, Any]:
        capture_dt: Optional[datetime] = None
        exif: Dict[str, Any] = {}
        prompt_exif: Dict[str, Any] = {}

        try:
            with Image.open(image_path) as img:
                raw_exif = img.getexif()
                if raw_exif:
                    for key, value in raw_exif.items():
                        tag = ExifTags.TAGS.get(key, key)
                        exif[tag] = value
                        if tag in self._EXIF_DATETIME_KEYS and not capture_dt:
                            capture_dt = self._parse_exif_datetime(value)
                        if tag in self._PROMPT_KEYS:
                            prompt_exif[tag] = value
        except Exception as exc:
            logging.getLogger("metadata_extractor").warning(
                "Failed to read EXIF for %s: %s", image_path.name, exc
            )

        if not capture_dt:
            stat = image_path.stat()
            capture_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        capture_dt = capture_dt if capture_dt.tzinfo else capture_dt.replace(tzinfo=timezone.utc)
        capture_dt_utc = capture_dt.astimezone(timezone.utc)

        return {
            "capture_time": capture_dt_utc,
            "capture_time_iso": capture_dt_utc.isoformat(),
            "exif": exif,
            "prompt_exif": prompt_exif,
        }

    @staticmethod
    def _parse_exif_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, str):
            for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None


class EmbeddingService:
    """Generates vector embeddings for photos using CLIP vision model."""

    def __init__(self, model_name: str) -> None:
        self._logger = logging.getLogger("embedding_service")
        self._logger.info(f"Loading CLIP model: {model_name}")
        self._model = SentenceTransformer(model_name)
        # CLIP models don't return dimension via get_sentence_embedding_dimension()
        # We need to encode a dummy image to get the actual dimension
        try:
            from PIL import Image
            import numpy as np
            dummy_img = Image.new('RGB', (224, 224))
            test_emb = self._model.encode(dummy_img, convert_to_numpy=True)
            self._dimension = len(test_emb)
            self._logger.info(f"CLIP embedding dimension: {self._dimension}")
        except Exception as exc:
            self._logger.warning(f"Failed to detect dimension: {exc}, defaulting to 512")
            self._dimension = 512

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_image(self, image_path: Path, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Encode image pixels directly to vector (metadata stored separately in Qdrant)."""
        try:
            from PIL import Image
            img = Image.open(image_path)
            embeddings = self._model.encode(
                img,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)
        except Exception as exc:
            self._logger.warning(f"Failed to encode {image_path.name}: {exc}")
            # Fallback: return zero vector
            return np.zeros(self._dimension, dtype=np.float32)


class VectorIndex:
    """Stores and retrieves image embeddings using Qdrant."""

    def __init__(self, *, config: SorterConfig, embedding_dim: int) -> None:
        self._logger = logging.getLogger("vector_index")
        self._collection_name = config.qdrant_collection
        self._dimension = embedding_dim
        self._client: Optional[QdrantClient] = None
        try:
            self._client = QdrantClient(url=config.qdrant_url)
            self._ensure_collection()
        except Exception as exc:
            self._logger.warning(
                "Qdrant недоступен (%s). Векторный поиск будет отключен.", exc
            )
            self._client = None

    def _ensure_collection(self) -> None:
        """Создает коллекцию или пересоздает, если размерность векторов не совпадает."""
        assert self._client is not None
        collections = self._client.get_collections().collections
        existing = {c.name: c for c in collections}
        
        if self._collection_name in existing:
            # Проверяем размерность существующей коллекции
            collection_info = self._client.get_collection(self._collection_name)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != self._dimension:
                self._logger.warning(
                    f"Размерность векторов изменилась ({current_dim} -> {self._dimension}), "
                    f"пересоздаю коллекцию {self._collection_name}"
                )
                self._client.delete_collection(collection_name=self._collection_name)
            else:
                return
        
        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )
        self._logger.info(f"Создана коллекция {self._collection_name} с размерностью {self._dimension}")

    def upsert(
        self,
        *,
        image_path: Path,
        category: str,
        business_type: Optional[str],
        description: Optional[str],
        capture_time_iso: Optional[str],
        metadata: Optional[Dict[str, Any]],
        embedding: np.ndarray,
    ) -> int:
        if self._client is None:
            return 0
        point_id = int(hash(image_path.resolve()) & 0x7FFFFFFF)
        
        # ✅ ВСЁ в metadata (Qdrant best practice)
        payload = {
            "path": str(image_path.resolve()),
            "category": category or "unknown",
            "business_type": business_type or "",
            "capture_time": capture_time_iso or "",
            # Merge all metadata into flat structure
            "description": description or "",  # AI description
            **(metadata or {}),  # EXIF + другие метаданные
        }
        
        point = PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)
        self._client.upsert(collection_name=self._collection_name, points=[point])
        return point_id

    def search(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 3,
        score_threshold: float = 0.82,
    ) -> List[Dict[str, Any]]:
        if self._client is None:
            return []
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        matches: List[Dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            matches.append(
                {
                    "path": payload.get("path"),
                    "category": payload.get("category"),
                    "business_type": payload.get("business_type"),
                    "description": payload.get("description"),  # AI-generated description
                    "capture_time": payload.get("capture_time"),
                    "metadata": payload.get("metadata"),
                    "score": hit.score,
                }
            )
        return matches


class OllamaClient:
    def __init__(self, config: SorterConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("ollama_client")
    
    def call_with_image(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        *,
        expect_json: bool = True,
    ) -> Dict[str, Any]:
        """Универсальный вызов Ollama с произвольными промптами"""
        self._logger.debug(f"Calling Ollama for {image_path.name}")
        
        for attempt in range(1, self._config.retry_limit + 1):
            try:
                response: ChatResponse = chat(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                            "content": user_prompt,
                            "images": [str(image_path)],
                        },
                    ],
                    options={"temperature": self._config.temperature},
                    format="json" if expect_json else None,
                )
                
                content = response.message.content
                if expect_json:
                    parsed = self._parse_json_payload(content)
                    return parsed | {"_raw_response": content}
                else:
                    return {"response": content, "_raw_response": content}
                    
            except Exception as exc:
                self._logger.warning(
                    f"  ❌ Attempt {attempt}/{self._config.retry_limit} failed: {exc}"
                )
                if attempt >= self._config.retry_limit:
                    raise RuntimeError(
                        f"Failed after {self._config.retry_limit} attempts"
                    ) from exc
        
        raise RuntimeError("Failed to call Ollama")

    def analyze_photo(
        self,
        image_path: Path,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        neighbor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._logger.debug(f"Analyzing {image_path.name} with model {self._config.model}")

        context_prompt = self._build_context_prompt(metadata=metadata, neighbor=neighbor)

        for attempt in range(1, self._config.retry_limit + 1):
            try:
                response: ChatResponse = chat(
                    model=self._config.model,
                    messages=[
                {
                    "role": "system",
                            "content": self._build_system_prompt(),
                },
                {
                    "role": "user",
                            "content": self._build_user_prompt(context_prompt=context_prompt),
                            "images": [str(image_path)],
                        },
                    ],
                    options={"temperature": self._config.temperature},
                    format="json",
                )

                content = response.message.content
                parsed = self._parse_json_payload(content)
                self._logger.debug(f"  ✅ Success, category: {parsed.get('category')}")
                return parsed | {"_raw_response": content}

            except Exception as exc:
                self._logger.warning(
                    f"  ❌ Attempt {attempt}/{self._config.retry_limit} failed: {type(exc).__name__}: {exc}"
                )
                if attempt >= self._config.retry_limit:
                    raise RuntimeError(
                        f"Failed to analyze {image_path.name} after {self._config.retry_limit} attempts"
                    ) from exc

        raise RuntimeError(f"Failed to analyze {image_path.name}")

    def _build_context_prompt(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        neighbor: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines: List[str] = []
        if metadata:
            capture_time = metadata.get("capture_time_iso")
            prompt_exif = metadata.get("prompt_exif") or {}
            if capture_time:
                lines.append(f"Captured at UTC: {capture_time}")
            if prompt_exif:
                exif_parts = [f"{key}={value}" for key, value in prompt_exif.items()]
                lines.append("EXIF hints: " + ", ".join(exif_parts))
        if neighbor:
            lines.append(
                "Nearest archived photo (vector & time similarity): "
                f"category={neighbor.get('category')}, "
                f"business_type={neighbor.get('business_type') or 'general'}, "
                f"description={neighbor.get('description', '')}, "
                f"capture_time={neighbor.get('capture_time')}"
            )
            lines.append(
                "Consider if current photo is from the same scene/context, "
                "but follow the main decision tree (screenshots/outdoor/home → sightseeing first)."
            )
        return "\n".join(lines)

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are an expert photo classifier. Your task: classify each photo as BUSINESS or SIGHTSEEING.\n\n"
            "OUTPUT FORMAT (strict JSON):\n"
            "{\n"
            '  "category": "business" | "sightseeing",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "summary": "brief description",\n'
            '  "business_insights": {"type": "slides"|"contacts"|"startups"|"receipts"|"general", "description": "..."},\n'
            '  "sightseeing_details": {"type": "cityscape"|"landmark"|"maps"|"home"|"other", "description": "..."}\n'
            "}\n\n"
            "CLASSIFICATION RULES:\n"
            "1. BUSINESS = work/travel related:\n"
            "   • slides: graphs, charts, presentation screens\n"
            "   • contacts: person profile with name/title\n"
            "   • startups: exhibition booth, branded stand\n"
            "   • receipts: payment receipts, boarding passes, tickets\n"
            "   • general: conference audience, venue\n\n"
            "2. SIGHTSEEING = leisure/tourism:\n"
            "   • cityscape: outdoor streets, buildings, skyline\n"
            "   • landmark: tourist attractions (Dubai Frame, Burj Khalifa, monuments)\n"
            "   • maps: navigation app screenshots (Google Maps, etc)\n"
            "   • home: pets, living room, residential\n"
            "   • other: food, restaurants, misc\n\n"
            "KEY DISTINCTIONS:\n"
            "  • Receipts/tickets/boarding passes → business/receipts\n"
            "  • Famous landmarks/monuments → sightseeing/landmark\n"
            "  • Map screenshots → sightseeing/maps\n"
            "  • Conference graphs → business/slides\n"
        )

    def _build_user_prompt(self, *, context_prompt: str = "") -> str:
        base = (
            "Classify this photo using the FEW-SHOT EXAMPLES below as a guide.\n\n"
            "═══════════════════════════════════════════════════════════════════\n"
            "FEW-SHOT EXAMPLES (learn from these):\n"
            "═══════════════════════════════════════════════════════════════════\n\n"
            "EXAMPLE 1: business/slides\n"
            "Image: Screen displaying a bar chart comparing \"China vs US\"\n"
            "Output:\n"
            "{\n"
            '  "category": "business",\n'
            '  "confidence": 0.95,\n'
            '  "summary": "Presentation slide showing economic comparison graph",\n'
            '  "business_insights": {"type": "slides", "description": "Graph on presentation screen"}\n'
            "}\n\n"
            "EXAMPLE 2: business/receipts\n"
            "Image: Aeroflot payment receipt with price 7600.00₽\n"
            "Output:\n"
            "{\n"
            '  "category": "business",\n'
            '  "confidence": 0.96,\n'
            '  "summary": "Aeroflot payment receipt for business travel",\n'
            '  "business_insights": {"type": "receipts", "description": "Payment receipt with text and numbers"}\n'
            "}\n\n"
            "EXAMPLE 3: business/contacts\n"
            "Image: TV screen showing Amanda Drury with name overlay \"CNBC Anchor\"\n"
            "Output:\n"
            "{\n"
            '  "category": "business",\n'
            '  "confidence": 0.92,\n'
            '  "summary": "Profile of Amanda Drury identified as CNBC anchor",\n'
            '  "business_insights": {"type": "contacts", "description": "Person profile with name and title"}\n'
            "}\n\n"
            "EXAMPLE 4: business/startups\n"
            "Image: Exhibition booth with \"DARKW3B\" branding and product displays\n"
            "Output:\n"
            "{\n"
            '  "category": "business",\n'
            '  "confidence": 0.94,\n'
            '  "summary": "DARKW3B exhibition booth at trade show",\n'
            '  "business_insights": {"type": "startups", "description": "Branded expo booth with demo"}\n'
            "}\n\n"
            "EXAMPLE 5: sightseeing/landmark\n"
            "Image: Dubai Frame - large golden frame structure outdoors\n"
            "Output:\n"
            "{\n"
            '  "category": "sightseeing",\n'
            '  "confidence": 0.98,\n'
            '  "summary": "Dubai Frame landmark with golden decorative structure",\n'
            '  "sightseeing_details": {"type": "landmark", "description": "Famous tourist monument"}\n'
            "}\n\n"
            "EXAMPLE 6: sightseeing/maps\n"
            "Image: Google Maps / navigation app with map view, pins, routes\n"
            "Output:\n"
            "{\n"
            '  "category": "sightseeing",\n'
            '  "confidence": 0.99,\n'
            '  "summary": "Navigation app showing map with route markers",\n'
            '  "sightseeing_details": {"type": "maps", "description": "Digital map interface with locations"}\n'
            "}\n\n"
            "EXAMPLE 7: sightseeing/cityscape\n"
            "Image: Sheikh Zayed Road with skyscrapers, cars, outdoor daytime\n"
            "Output:\n"
            "{\n"
            '  "category": "sightseeing",\n'
            '  "confidence": 0.98,\n'
            '  "summary": "Dubai highway with modern skyscrapers",\n'
            '  "sightseeing_details": {"type": "cityscape", "description": "Outdoor urban street view"}\n'
            "}\n\n"
            "EXAMPLE 8: sightseeing/home\n"
            "Image: Cat sitting on carpet in living room\n"
            "Output:\n"
            "{\n"
            '  "category": "sightseeing",\n'
            '  "confidence": 0.97,\n'
            '  "summary": "Pet cat on home carpet",\n'
            '  "sightseeing_details": {"type": "home", "description": "Domestic pet photo"}\n'
            "}\n\n"
            "═══════════════════════════════════════════════════════════════════\n"
            "NOW CLASSIFY THE PROVIDED PHOTO:\n"
            "═══════════════════════════════════════════════════════════════════\n"
            "Use the same pattern as examples above. Match structure exactly.\n\n"
            "KEY RULES:\n"
            "• Map interface (Google Maps, navigation, pins, routes) → sightseeing/maps (NOT slides!)\n"
            "• Payment receipt / boarding pass / ticket → business/receipts\n"
            "• Famous landmark (Dubai Frame, Burj Khalifa) → sightseeing/landmark\n"
            "• Graph/chart on presentation screen → business/slides\n"
            "• Exhibition booth → business/startups\n"
            "• Profile with name → business/contacts\n\n"
            "⚠️ CRITICAL: Map apps have text/UI but are NEVER business/slides!\n"
        )
        if context_prompt:
            base += f"\n\nADDITIONAL CONTEXT:\n{context_prompt}\n"
        return base

    @staticmethod
    def _parse_json_payload(payload: str) -> Dict[str, Any]:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Assistant response did not contain JSON payload.")
        return json.loads(payload[start : end + 1])


class PhotoRepository:
    def __init__(self, database_path: Path) -> None:
        self._db_path = database_path
        self._connection = sqlite3.connect(str(database_path))
        self._connection.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        # Таблица категорий (динамические, создаются AI)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                parent_id INTEGER,
                description TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES categories(id)
            )
            """
        )
        
        # Таблица анализа фото
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS photo_analysis (
                id INTEGER PRIMARY KEY,
                original_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                category_id INTEGER NOT NULL,
                subcategory_id INTEGER,
                confidence REAL,
                destination_path TEXT,
                raw_json TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(id),
                FOREIGN KEY (subcategory_id) REFERENCES categories(id)
            )
            """
        )
        
        # Лог вызовов AI (для отладки)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_calls (
                id INTEGER PRIMARY KEY,
                photo_path TEXT NOT NULL,
                call_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        
        self._connection.commit()
        
        # Инициализируем базовые категории, если БД пустая
        self._init_base_categories()
    
    def _init_base_categories(self) -> None:
        """Создаёт только топ-level категории при первом запуске"""
        cur = self._connection.execute("SELECT COUNT(*) FROM categories")
        if cur.fetchone()[0] == 0:
            now = datetime.now(timezone.utc).isoformat()
            base_categories = [
                ("business", "Work-related: conferences, documents, receipts"),
                ("sightseeing", "Tourism: landmarks, cityscapes, maps"),
                ("personal", "Home: pets, family, daily life"),
            ]
            for name, desc in base_categories:
                self._connection.execute(
                    "INSERT INTO categories (name, parent_id, description, created_at) VALUES (?, NULL, ?, ?)",
                    (name, desc, now)
                )
            self._connection.commit()
    
    def get_top_categories(self) -> List[Dict[str, Any]]:
        """Возвращает все top-level категории"""
        cur = self._connection.execute(
            "SELECT id, name, description FROM categories WHERE parent_id IS NULL ORDER BY name"
        )
        return [{"id": row[0], "name": row[1], "description": row[2]} for row in cur.fetchall()]
    
    def get_subcategories(self, parent_id: int) -> List[Dict[str, Any]]:
        """Возвращает все подкатегории для заданной категории"""
        cur = self._connection.execute(
            "SELECT id, name, description FROM categories WHERE parent_id = ? ORDER BY name",
            (parent_id,)
        )
        return [{"id": row[0], "name": row[1], "description": row[2]} for row in cur.fetchall()]
    
    def find_or_create_category(self, name: str, parent_id: Optional[int], description: str) -> int:
        """Находит существующую или создаёт новую категорию"""
        # Проверяем существование
        cur = self._connection.execute(
            "SELECT id FROM categories WHERE name = ? AND parent_id IS ?",
            (name, parent_id)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        
        # Создаём новую
        now = datetime.now(timezone.utc).isoformat()
        cur = self._connection.execute(
            "INSERT INTO categories (name, parent_id, description, created_at) VALUES (?, ?, ?, ?)",
            (name, parent_id, description, now)
        )
        self._connection.commit()
        return cur.lastrowid
    
    def log_ai_call(self, photo_path: Path, call_type: str, prompt: str, response: str) -> None:
        """Логирует вызов AI для отладки"""
        now = datetime.now(timezone.utc).isoformat()
        self._connection.execute(
            "INSERT INTO ai_calls (photo_path, call_type, prompt, response, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(photo_path), call_type, prompt, response, now)
        )
        self._connection.commit()

    def is_processed(self, image_path: Path) -> bool:
        cur = self._connection.execute(
            "SELECT 1 FROM photo_analysis WHERE original_path = ?", (str(image_path),)
        )
        return bool(cur.fetchone())

    def upsert_result(
        self,
        image_path: Path,
        destination_path: Optional[Path],
        category_id: int,
        subcategory_id: Optional[int],
        confidence: float,
        raw_json: str,
    ) -> None:
        processed_at = datetime.now(timezone.utc).isoformat()

        self._connection.execute(
            """
            INSERT INTO photo_analysis (
                original_path,
                file_name,
                category_id,
                subcategory_id,
                confidence,
                destination_path,
                raw_json,
                processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(original_path) DO UPDATE SET
                category_id=excluded.category_id,
                subcategory_id=excluded.subcategory_id,
                confidence=excluded.confidence,
                destination_path=excluded.destination_path,
                raw_json=excluded.raw_json,
                processed_at=excluded.processed_at
            """,
            (
                str(image_path),
                image_path.name,
                category_id,
                subcategory_id,
                confidence,
                str(destination_path) if destination_path else None,
                raw_json,
                processed_at,
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


@dataclass
class ClassificationResult:
    """Результат интеллектуальной классификации"""
    category_id: int
    category_name: str
    subcategory_id: Optional[int]
    subcategory_name: Optional[str]
    confidence: float
    description: str  # Детальное AI-описание картинки
    raw_data: Dict[str, Any]


class IntelligentClassifier:
    """Мета-система AI классификации с динамическими категориями"""
    
    def __init__(
        self,
        ollama: OllamaClient,
        repository: "PhotoRepository",
        embedder: EmbeddingService,
        vector_index: VectorIndex,
    ) -> None:
        self._ollama = ollama
        self._repo = repository
        self._embedder = embedder
        self._vector_index = vector_index
        self._logger = logging.getLogger("intelligent_classifier")
    
    def classify(self, image_path: Path, metadata: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Многоэтапная классификация:
        1. STEP 1: Определение top-level категории (business/sightseeing/personal)
        2. STEP 2: Векторный поиск похожих фото
        3. STEP 3: Определение подкатегории (использует существующие или создаёт новую)
        """
        # Цветной вывод
        BLUE, GREEN, YELLOW, CYAN, MAGENTA, RESET, BOLD = (
            '\033[94m', '\033[92m', '\033[93m', '\033[96m', '\033[95m', '\033[0m', '\033[1m'
        )
        
        self._logger.info(f"{CYAN}{'─'*70}{RESET}")
        self._logger.info(f"{BOLD}{BLUE}🧠 INTELLIGENT CLASSIFICATION: {image_path.name}{RESET}")
        self._logger.info(f"{CYAN}{'─'*70}{RESET}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 0: GENERATE DETAILED DESCRIPTION
        # ═══════════════════════════════════════════════════════════
        self._logger.info(f"{MAGENTA}📝 STEP 0: Generating detailed description...{RESET}")
        
        desc_system = "You are a professional photo analyst. Describe the image in detail."
        desc_user = (
            "Analyze this image and provide a detailed, objective description. "
            "Include:\n"
            "- Main subjects and objects\n"
            "- Setting and environment\n"
            "- Colors and composition\n"
            "- Any text or notable details\n"
            "- Overall mood or purpose\n\n"
            "Keep it factual and comprehensive (2-4 sentences)."
        )
        
        desc_result = self._ollama.call_with_image(image_path, desc_system, desc_user)
        
        # Извлекаем description (модель может вернуть JSON)
        raw_desc = desc_result.get("description", desc_result.get("_raw_response", "No description"))
        if isinstance(raw_desc, str) and raw_desc.strip().startswith("{"):
            # Парсим JSON если модель вернула структуру
            try:
                parsed = json.loads(raw_desc)
                description = parsed.get("analysis", parsed.get("description", raw_desc))
            except json.JSONDecodeError:
                description = raw_desc
        else:
            description = raw_desc
        
        self._logger.info(f"{GREEN}   → Description: {description[:100]}...{RESET}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: TOP-LEVEL CATEGORY
        # ═══════════════════════════════════════════════════════════
        self._logger.info(f"{MAGENTA}📋 STEP 1: Determining top-level category...{RESET}")
        
        top_categories = self._repo.get_top_categories()
        step1_system, step1_user = self._build_step1_prompts(top_categories)
        
        step1_result = self._ollama.call_with_image(image_path, step1_system, step1_user)
        self._repo.log_ai_call(image_path, "step1_category", step1_user, step1_result.get("_raw_response", ""))
        
        category_name = step1_result.get("category", "unknown")
        category_id = next((c["id"] for c in top_categories if c["name"] == category_name), None)
        
        if not category_id:
            self._logger.warning(f"{YELLOW}⚠️  Unknown category '{category_name}', falling back to 'sightseeing'{RESET}")
            category_name = "sightseeing"
            category_id = next(c["id"] for c in top_categories if c["name"] == "sightseeing")
        
        self._logger.info(f"{GREEN}   → Category: {category_name} (id={category_id}){RESET}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: VECTOR SEARCH FOR SIMILAR PHOTOS
        # ═══════════════════════════════════════════════════════════
        self._logger.info(f"{MAGENTA}🔍 STEP 2: Searching for similar photos...{RESET}")
        
        embedding = self._embedder.encode_image(image_path, metadata)
        similar_matches = self._vector_index.search(embedding, top_k=3, score_threshold=0.75)
        
        similar_context = None
        if similar_matches:
            similar = similar_matches[0]
            similar_context = {
                "description": similar.get("description", ""),
                "category": similar.get("category", ""),
                "business_type": similar.get("business_type", ""),
                "score": similar.get("score", 0.0),
            }
            self._logger.info(
                f"{GREEN}   → Found similar: {similar.get('category')}/{similar.get('business_type')} "
                f"(score: {similar.get('score', 0):.3f}){RESET}"
            )
        else:
            self._logger.info(f"{YELLOW}   → No similar photos found{RESET}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: SUBCATEGORY DETERMINATION
        # ═══════════════════════════════════════════════════════════
        self._logger.info(f"{MAGENTA}🎯 STEP 3: Determining subcategory...{RESET}")
        
        existing_subcats = self._repo.get_subcategories(category_id)
        step2_system, step2_user = self._build_step2_prompts(
            category_name=category_name,
            existing_subcategories=existing_subcats,
            similar_context=similar_context
        )
        
        step2_result = self._ollama.call_with_image(image_path, step2_system, step2_user)
        self._repo.log_ai_call(image_path, "step2_subcategory", step2_user, step2_result.get("_raw_response", ""))
        
        subcategory_name = step2_result.get("subcategory", "other")
        subcategory_desc = step2_result.get("subcategory_description", "")
        is_new = step2_result.get("is_new_subcategory", False)
        
        subcategory_id = self._repo.find_or_create_category(
            name=subcategory_name,
            parent_id=category_id,
            description=subcategory_desc
        )
        
        if is_new:
            self._logger.info(f"{GREEN}{BOLD}   → ✨ Created NEW subcategory: {subcategory_name}{RESET}")
        else:
            self._logger.info(f"{GREEN}   → Using existing: {subcategory_name} (id={subcategory_id}){RESET}")
        
        confidence = step2_result.get("confidence", 0.8)
        
        self._logger.info(f"{CYAN}{'─'*70}{RESET}")
        self._logger.info(
            f"{BOLD}{GREEN}✅ RESULT: {category_name}/{subcategory_name} (confidence: {confidence:.2f}){RESET}"
        )
        self._logger.info(f"{CYAN}{'─'*70}{RESET}\n")
        
        return ClassificationResult(
            category_id=category_id,
            category_name=category_name,
            subcategory_id=subcategory_id,
            subcategory_name=subcategory_name,
            confidence=confidence,
            description=description,
            raw_data={"step1": step1_result, "step2": step2_result}
        )
    
    def _build_step1_prompts(self, top_categories: List[Dict[str, Any]]) -> tuple[str, str]:
        """Строит промпты для STEP 1: определение top-level категории"""
        categories_list = "\n".join(
            f"  • {cat['name']}: {cat['description']}"
            for cat in top_categories
        )
        
        system_prompt = f"""You are an expert photo classifier. 
Your task: classify the photo into ONE of the following top-level categories:

{categories_list}

Return JSON:
{{
  "category": "category_name",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}}"""
        
        user_prompt = f"""Analyze this photo and determine which top-level category it belongs to.

Available categories:
{categories_list}

Choose the MOST appropriate category based on the photo's content and context.
Return your decision in JSON format."""
        
        return system_prompt, user_prompt
    
    def _build_step2_prompts(
        self,
        category_name: str,
        existing_subcategories: List[Dict[str, Any]],
        similar_context: Optional[Dict[str, Any]]
    ) -> tuple[str, str]:
        """Строит промпты для STEP 2: определение подкатегории"""
        
        if existing_subcategories:
            subcats_list = "\n".join(
                f"  • {sub['name']}: {sub.get('description', 'no description')}"
                for sub in existing_subcategories
            )
            subcats_section = f"""
EXISTING SUBCATEGORIES in '{category_name}':
{subcats_list}

You can either:
1. Choose one of the existing subcategories if it fits well
2. Create a NEW subcategory if none of the existing ones are appropriate
"""
        else:
            subcats_section = f"""
No subcategories exist yet for '{category_name}'.
You should CREATE a new subcategory name that best describes this photo.
"""
        
        similar_section = ""
        if similar_context:
            similar_section = f"""
SIMILAR PHOTO FOUND:
  • Category: {similar_context['category']}/{similar_context['business_type']}
  • Description: {similar_context['description']}
  • Similarity score: {similar_context['score']:.3f}

Consider if the current photo belongs to the same subcategory as the similar one.
"""
        
        system_prompt = f"""You are an expert at creating a photo taxonomy.
You're now refining the classification within the '{category_name}' category.

{subcats_section}

Return JSON:
{{
  "subcategory": "subcategory_name",
  "subcategory_description": "brief description (if new)",
  "is_new_subcategory": true/false,
  "reasoning": "explanation",
  "confidence": 0.0-1.0
}}

RULES:
• Use existing subcategories when possible
• Create new ones ONLY if necessary
• New subcategory names should be short (1-2 words), lowercase, descriptive
• Examples of good names: "slides", "receipts", "landmarks", "maps", "contacts"
"""
        
        user_prompt = f"""The photo has been classified as: {category_name}

{subcats_section}
{similar_section}

Analyze the photo and decide:
1. Which existing subcategory fits best? OR
2. Should we create a new subcategory? (give it a name and description)

Return your decision in JSON format."""
        
        return system_prompt, user_prompt


class PhotoSorter:
    def __init__(
        self,
        config: SorterConfig,
        *,
        dry_run: bool = False,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self._config = config
        self._dry_run = dry_run
        self._force = force
        self._limit = limit
        self._ollama = OllamaClient(config)
        self._repository = PhotoRepository(config.database)
        self._metadata = MetadataExtractor()
        self._embedder = EmbeddingService(config.embedding_model)
        self._vector_index = VectorIndex(config=config, embedding_dim=self._embedder.dimension)
        self._classifier = IntelligentClassifier(
            ollama=self._ollama,
            repository=self._repository,
            embedder=self._embedder,
            vector_index=self._vector_index,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("photo_sorter")
        self._configure_logging()
        self._validate_paths()

    def _compose_description(self, analysis: Dict[str, Any]) -> str:
        category = (analysis.get("category") or "unknown").lower()
        parts: List[str] = [f"category={category}"]
        biz = analysis.get("business_insights") or {}
        sight = analysis.get("sightseeing_details") or {}

        if category == "business":
            biz_type = (biz.get("type") or "general").lower()
            parts.append(f"type={biz_type}")
            summary = biz.get("summary")
            if summary:
                parts.append(summary)
            slide_description = biz.get("slide_description")
            if slide_description:
                parts.append(f"Slide detail: {slide_description}")
            contacts = biz.get("contacts") or []
            if contacts:
                contact_bits = []
                for contact in contacts:
                    fields = [contact.get("name"), contact.get("role"), contact.get("company")]
                    contact_bits.append(
                        ", ".join(filter(None, fields)) or "contact"
                    )
                parts.append("Contacts: " + "; ".join(contact_bits))
            startups = biz.get("startup_profiles") or []
            if startups:
                startup_bits = []
                for profile in startups:
                    fields = [profile.get("name"), profile.get("focus"), profile.get("website")]
                    startup_bits.append(
                        ", ".join(filter(None, fields)) or "startup"
                    )
                parts.append("Startups: " + "; ".join(startup_bits))
        else:
            location = sight.get("location_guess")
            if location:
                parts.append(f"Location guess: {location}")
            visuals = sight.get("visual_elements") or []
            if visuals:
                parts.append("Visual elements: " + ", ".join(visuals[:6]))
            vibe = sight.get("vibe")
            if vibe:
                parts.append(f"Vibe: {vibe}")

        notable = analysis.get("notable_text") or []
        if notable:
            parts.append("Notable text: " + ", ".join(notable[:6]))

        safety = analysis.get("safety_checks") or {}
        if safety.get("contains_sensitive_info"):
            parts.append("Sensitive info present")

        return " ".join(parts)

    def _compact_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает EXIF метаданные в flat structure для Qdrant payload."""
        prompt_exif = metadata.get("prompt_exif") or {}
        # Flat structure - все ключи на верхнем уровне для лучшего поиска
        result = {"capture_time_iso": metadata.get("capture_time_iso")}
        result.update(prompt_exif)  # Make, Model, DateTime и т.д.
        return result

    def run(self) -> None:
        # ANSI color codes
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        images = self._gather_images(self._config.source_dir)
        processed = 0
        for image_path in images:
            if self._limit is not None and processed >= self._limit:
                break
            skip = self._repository.is_processed(image_path) and not self._force
            
            if skip:
                self._logger.info(f"{YELLOW}⏩ [skip] {image_path.name} (cached){RESET}")
                continue
            
            self._logger.info(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
            self._logger.info(f"{BOLD}{BLUE}📸 Processing: {image_path.name}{RESET}")
            self._logger.info(f"{CYAN}{'='*70}{RESET}")
            
            try:
                # Extract metadata
                metadata = self._metadata.extract(image_path)
                if metadata.get("capture_time_iso"):
                    self._logger.info(f"{MAGENTA}📅 Captured: {metadata['capture_time_iso']}{RESET}")
                
                # 🧠 INTELLIGENT CLASSIFICATION (multi-step AI)
                result = self._classifier.classify(image_path, metadata)
                
                # Generate embedding and store in Qdrant
                embedding = self._embedder.encode_image(image_path, metadata)
                metadata_payload = self._compact_metadata(metadata)
                
                self._vector_index.upsert(
                    image_path=image_path,
                    category=result.category_name,
                    business_type=result.subcategory_name,
                    description=result.description,  # ✅ AI-описание в metadata
                    capture_time_iso=metadata.get("capture_time_iso"),
                    metadata=metadata_payload,
                    embedding=embedding,
                )
                
                # Move file
                destination = (
                    self._resolve_destination(image_path, result.category_name, result.subcategory_name)
                    if not self._dry_run
                    else None
                )
                final_target = None
                if destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    final_target = self._deduplicate_path(destination)
                    self._move_file(image_path, final_target)
                    rel_path = final_target.relative_to(self._config.output_dir)
                    self._logger.info(f"{GREEN}✅ Moved to: {rel_path}{RESET}")
                
                # Save to database
                self._repository.upsert_result(
                    image_path=image_path,
                    destination_path=final_target,
                    category_id=result.category_id,
                    subcategory_id=result.subcategory_id,
                    confidence=result.confidence,
                    raw_json=json.dumps(result.raw_data, separators=(",", ":")),
                )
                processed += 1
                
            except Exception as exc:
                self._logger.error(f"{RED}❌ Failed to process {image_path.name}: {exc}{RESET}", exc_info=True)

    def close(self) -> None:
        self._repository.close()

    @staticmethod
    def _gather_images(source_dir: Path) -> Iterable[Path]:
        candidates = sorted(
            path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        return candidates

    def _resolve_destination(
        self,
        image_path: Path,
        category_name: str,
        subcategory_name: Optional[str]
    ) -> Path:
        """Определяет путь назначения на основе категории и подкатегории"""
        safe_category = category_name.lower().replace(" ", "_")
        safe_subcat = (subcategory_name or "other").lower().replace(" ", "_")
        
        target_dir = self._config.output_dir / safe_category / safe_subcat
        return target_dir / image_path.name

    def _deduplicate_path(self, target_path: Path) -> Path:
        if not target_path.exists():
            return target_path
        stem, suffix = target_path.stem, target_path.suffix
        counter = 1
        while True:
            candidate = target_path.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _move_file(self, source: Path, destination: Path) -> None:
        try:
            source.rename(destination)
        except OSError:
            import shutil

            shutil.move(str(source), str(destination))

    def _configure_logging(self) -> None:
        pass  # Logging configured in main()

    def _validate_paths(self) -> None:
        if not self._config.source_dir.exists() or not self._config.source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self._config.source_dir}")
        if not self._config.output_dir.exists():
            self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_similar_entry(self, image_path: Path, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        embedding = self._embedder.encode_image(image_path, metadata)
        matches = self._vector_index.search(embedding, top_k=3, score_threshold=0.82)
        if not matches:
            return None

        capture_time = metadata.get("capture_time")
        if capture_time and isinstance(capture_time, datetime):
            for match in matches:
                match_time_iso = match.get("capture_time")
                if not match_time_iso:
                    continue
                try:
                    match_time = datetime.fromisoformat(match_time_iso)
                except ValueError:
                    continue
                delta = abs(capture_time - match_time)
                if delta <= timedelta(seconds=15):
                    return match
        return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and sort conference photos with a local Ollama model."
    )
    parser.add_argument(
        "--source",
        default="img",
        help="Directory containing the source photos.",
    )
    parser.add_argument(
        "--destination",
        default="sorted_photos",
        help="Root directory for sorted outputs.",
    )
    parser.add_argument(
        "--database",
        default="photo_analysis.sqlite",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--model",
        default="gemma3:27b-it-qat",
        help="Ollama model identifier to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (0.0 = deterministic).",
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=3,
        help="How many times to retry a failed Ollama call.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=3.0,
        help="Seconds to back off between retries (multiplied by attempt index).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without moving files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze files even if they already exist in the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of photos to process this run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = SorterConfig(
        source_dir=Path(args.source).resolve(),
        output_dir=Path(args.destination).resolve(),
        database=Path(args.database).resolve(),
        model=args.model,
        temperature=args.temperature,
        retry_limit=args.retry_limit,
        retry_backoff_seconds=args.retry_backoff,
    )
    sorter = PhotoSorter(
        config,
        dry_run=args.dry_run,
        force=args.force,
        limit=args.limit,
    )
    try:
        sorter.run()
    finally:
        sorter.close()


if __name__ == "__main__":
    main()
