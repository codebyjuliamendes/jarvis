from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import httpx

from . import security
from .config import Settings
from .documents import ExtractionError, extract_text

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".cache",
    ".local",
    ".config",
    ".ssh",
    ".gnupg",
    ".mozilla",
    ".npm",
    ".cargo",
    ".rustup",
    "snap",
    "data",
    "backups",
    "jarvis-backups",
}

MAX_FILE_BYTES = 12 * 1024 * 1024
EMBED_BATCH_SIZE = 16


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    text = text.replace("\x00", " ").replace("\r\n", "\n").strip()
    if not text:
        return []
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 2 <= size:
            current = (current + "\n\n" + paragraph).strip()
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > size:
            chunks.append(paragraph[:size].strip())
            paragraph = paragraph[max(0, size - overlap) :].strip()
        current = paragraph
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk) >= 20]


class RagIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = Path(settings.db_path).parent / "rag.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._db_lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "path TEXT UNIQUE NOT NULL, "
                "mtime REAL NOT NULL, "
                "size INTEGER NOT NULL, "
                "indexed_at TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS chunks ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "document_id INTEGER NOT NULL, "
                "chunk_index INTEGER NOT NULL, "
                "text TEXT NOT NULL, "
                "embedding TEXT NOT NULL, "
                "FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)"
            )
            self._db.commit()

    def _embed(self, text: str) -> List[float]:
        try:
            response = httpx.post(
                self.settings.ollama_url + "/api/embeddings",
                json={"model": self.settings.rag_model, "prompt": text},
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError("Falha ao falar com o Ollama: {}".format(exc)) from exc
        if response.status_code == 404:
            raise RuntimeError(
                "Modelo de embeddings '{}' não encontrado. Rode: ollama pull {}".format(
                    self.settings.rag_model, self.settings.rag_model
                )
            )
        if response.status_code >= 400:
            raise RuntimeError(
                "Ollama retornou {}: {}".format(response.status_code, response.text[:300])
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Resposta inválida do Ollama") from exc
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Ollama não retornou um embedding válido")
        return [float(value) for value in embedding]

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                self.settings.ollama_url + "/api/embed",
                json={
                    "model": self.settings.rag_model,
                    "input": list(texts),
                },
                timeout=300.0,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError("Falha ao falar com o Ollama: {}".format(exc)) from exc
        if response.status_code == 404:
            return [self._embed(text) for text in texts]
        if response.status_code >= 400:
            raise RuntimeError(
                "Ollama retornou {}: {}".format(response.status_code, response.text[:300])
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Resposta inválida do Ollama") from exc
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            return [self._embed(text) for text in texts]
        result: List[List[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                return [self._embed(text) for text in texts]
            result.append([float(value) for value in embedding])
        return result

    def _extensions(self) -> set:
        return {
            item.strip().lower()
            for item in self.settings.rag_extensions.replace(";", ",").split(",")
            if item.strip()
        }

    def supports(self, path: Path) -> bool:
        return Path(path).suffix.lower() in self._extensions()

    def _iter_files(self, roots: Sequence[Path]) -> Iterable[Path]:
        extensions = self._extensions()
        for root in roots:
            root = Path(root).expanduser().resolve(strict=False)
            if root.is_file():
                candidates = [root]
            elif root.is_dir():
                candidates = root.rglob("*")
            else:
                continue
            for path in candidates:
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    if path.suffix.lower() not in extensions:
                        continue
                    if any(part in SKIP_DIRS for part in path.parts):
                        continue
                    if security.is_sensitive(path):
                        continue
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                yield path

    def _existing(self, path: str):
        with self._db_lock:
            return self._db.execute(
                "SELECT id, mtime, size FROM documents WHERE path = ?", (path,)
            ).fetchone()

    def _delete_document(self, path: str) -> None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT id FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return
            self._db.execute("DELETE FROM chunks WHERE document_id = ?", (row["id"],))
            self._db.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            self._db.commit()

    def _store_document(
        self,
        path: str,
        mtime: float,
        size: int,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT id FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if row is not None:
                self._db.execute(
                    "DELETE FROM chunks WHERE document_id = ?", (row["id"],)
                )
                self._db.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            cursor = self._db.execute(
                "INSERT INTO documents (path, mtime, size, indexed_at) VALUES (?, ?, ?, ?)",
                (path, mtime, size, datetime.now().astimezone().isoformat()),
            )
            document_id = cursor.lastrowid
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self._db.execute(
                    "INSERT INTO chunks (document_id, chunk_index, text, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (document_id, index, chunk, json.dumps(list(embedding))),
                )
            self._db.commit()

    def _prune_missing(self) -> int:
        with self._db_lock:
            rows = self._db.execute("SELECT path FROM documents").fetchall()
        removed = 0
        for row in rows:
            path = Path(row["path"])
            try:
                stale = not path.is_file() or security.is_sensitive(path)
            except OSError:
                stale = True
            if stale:
                self._delete_document(row["path"])
                removed += 1
                logger.info("Removido do índice (arquivo ausente/sensível): %s", path)
        return removed

    def index(
        self,
        paths: Optional[Sequence[Path]] = None,
        force: bool = False,
        max_files: int = 400,
        max_chunks: int = 4000,
    ) -> str:
        roots = list(paths) if paths else list(self.settings.rag_roots)
        if not roots:
            return "Nenhuma pasta configurada para indexar (JARVIS_RAG_ROOTS)."
        if not self._index_lock.acquire(blocking=False):
            return "Já existe uma indexação em andamento."
        files_seen = 0
        files_indexed = 0
        chunks_total = 0
        skipped = 0
        errors = 0
        truncated = False
        try:
            for path in self._iter_files(roots):
                if files_indexed >= max_files or chunks_total >= max_chunks:
                    truncated = True
                    break
                files_seen += 1
                path_str = str(path)
                try:
                    stat = path.stat()
                    existing = self._existing(path_str)
                    if (
                        not force
                        and existing is not None
                        and abs(existing["mtime"] - stat.st_mtime) < 0.001
                        and existing["size"] == stat.st_size
                    ):
                        skipped += 1
                        continue
                    text = extract_text(path)
                    chunks = chunk_text(text)
                    if not chunks:
                        self._delete_document(path_str)
                        skipped += 1
                        continue
                    remaining = max_chunks - chunks_total
                    if len(chunks) > remaining:
                        chunks = chunks[:remaining]
                        truncated = True
                    embeddings: List[List[float]] = []
                    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                        embeddings.extend(
                            self._embed_batch(chunks[start : start + EMBED_BATCH_SIZE])
                        )
                    self._store_document(
                        path_str, stat.st_mtime, stat.st_size, chunks, embeddings
                    )
                    files_indexed += 1
                    chunks_total += len(chunks)
                    logger.info("Indexado: %s (%s trechos)", path, len(chunks))
                except (ExtractionError, OSError) as exc:
                    errors += 1
                    logger.warning("Falha ao indexar %s: %s", path, exc)
                except RuntimeError:
                    errors += 1
                    raise
            lines = [
                "Indexação concluída.",
                "Arquivos vistos: {}".format(files_seen),
                "Arquivos indexados: {}".format(files_indexed),
                "Trechos gerados: {}".format(chunks_total),
                "Ignorados: {}".format(skipped),
                "Erros: {}".format(errors),
            ]
            if paths is None and not truncated:
                removed = self._prune_missing()
                if removed:
                    lines.append("Removidos do índice (ausentes): {}".format(removed))
            if truncated:
                lines.append("Limite atingido; rode novamente para continuar.")
            return "\n".join(lines)
        except RuntimeError as exc:
            return "Erro na indexação: {}".format(exc)
        finally:
            self._index_lock.release()

    def search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        query = (query or "").strip()
        if not query:
            return []
        try:
            import numpy as np
        except ImportError:
            return []
        try:
            query_vector = np.array(self._embed(query), dtype="float32")
        except RuntimeError as exc:
            logger.warning("Busca sem embeddings: %s", exc)
            return []
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return []
        with self._db_lock:
            rows = self._db.execute(
                "SELECT c.text, c.embedding, c.chunk_index, d.path "
                "FROM chunks c JOIN documents d ON d.id = c.document_id"
            ).fetchall()
        scored: List[Dict[str, object]] = []
        for row in rows:
            try:
                vector = np.array(json.loads(row["embedding"]), dtype="float32")
            except (ValueError, TypeError):
                continue
            norm = float(np.linalg.norm(vector))
            if norm == 0.0 or vector.shape != query_vector.shape:
                continue
            score = float(np.dot(query_vector, vector) / (query_norm * norm))
            scored.append(
                {
                    "path": row["path"],
                    "chunk": row["chunk_index"],
                    "score": score,
                    "text": row["text"],
                }
            )
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        return scored[: max(1, int(k))]

    def stats(self) -> Dict[str, int]:
        with self._db_lock:
            documents = self._db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": int(documents), "chunks": int(chunks)}

    def close(self) -> None:
        with self._db_lock:
            self._db.close()
