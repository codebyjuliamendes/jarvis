"""Memória local do Jarvis: indexação e busca nos arquivos da usuária.

Três modos de busca (``JARVIS_RAG_MODE``):

- ``vector``: similaridade de cosseno entre embeddings (Ollama). Usa a extensão
  sqlite-vec quando ela carrega; senão, NumPy sobre o JSON guardado na tabela.
- ``bm25``: busca lexical com FTS5 (BM25), sem chamar o Ollama.
- ``hybrid``: as duas, combinadas por Reciprocal Rank Fusion (RRF).

A tabela ``chunks`` é a fonte da verdade (texto + embedding em JSON). ``chunks_fts``
e ``chunks_vec`` são índices derivados, reconstruídos a partir dela quando faltam ou
divergem — índices antigos continuam funcionando sem reindexar.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

RAG_MODES = ("vector", "bm25", "hybrid")
# Padrão escolhido pela medição (evals/results/RAG.md): maior MRR na metade dev.
DEFAULT_MODE = "bm25"
VECTOR_BACKENDS = ("auto", "sqlite-vec", "numpy")
# Quantos candidatos cada perna entrega ao RRF no modo híbrido. Fixado antes de
# medir (ver docs/superpowers/specs/2026-11-08-rag-hibrido-design.md).
HYBRID_CANDIDATES = 20
DEFAULT_RRF_K = 60
# Prefixos com que o nomic-embed-text foi treinado.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Lista curta e genérica (artigos, preposições, pronomes, verbos auxiliares), escolhida
# antes de medir e não ajustada pelo dataset. Comparada sem acento.
STOPWORDS = frozenset(
    """
    a o as os um uma uns umas de do da dos das no na nos nas em ao aos à às pelo pela
    pelos pelas por para pra pro com sem sob sobre entre ate apos e ou mas nem que se
    como quando onde qual quais quem quanto quanta quantos quantas porque eu tu ele ela
    nos vos eles elas me te lhe lhes meu minha meus minhas seu sua seus suas nosso nossa
    esse essa isso este esta isto aquele aquela aquilo foi era ser estar esta estao
    tem ter ha vai vou sao fica ficou e é
    """.split()
)


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


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def fts_query(text: str) -> str:
    """Pergunta em linguagem natural -> consulta FTS5 (tokens entre aspas, unidos por OR).

    Tira stopwords e tokens de uma letra (dígitos ficam). Devolve "" se nada sobrar.
    """
    seen: List[str] = []
    for token in re.findall(r"\w+", (text or "").lower()):
        plain = _strip_accents(token)
        if plain in STOPWORDS or (len(plain) < 2 and not plain.isdigit()):
            continue
        if plain not in seen:
            seen.append(plain)
    return " OR ".join('"{}"'.format(token.replace('"', "")) for token in seen)


def rrf_fuse(
    rankings: Sequence[Sequence[int]], k: int = DEFAULT_RRF_K
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = soma de 1 / (k + posição de d em cada lista).

    Posições começam em 1. Empate: vale a ordem de chegada (a primeira lista primeiro).
    """
    scores: Dict[int, float] = {}
    order: Dict[int, int] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
            if item not in order:
                order[item] = len(order)
    return sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))


def use_prefixes(setting: str, model: str) -> bool:
    value = (setting or "auto").strip().lower()
    if value in {"1", "true", "yes", "on", "sim"}:
        return True
    if value in {"0", "false", "no", "off", "nao", "não"}:
        return False
    return "nomic" in (model or "").lower()


def _vec_blob(vector: Sequence[float]) -> bytes:
    return struct.pack("<{}f".format(len(vector)), *vector)


class RagIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        mode = (settings.rag_mode or DEFAULT_MODE).strip().lower()
        if mode not in RAG_MODES:
            logger.warning("JARVIS_RAG_MODE inválido (%r); usando %r.", mode, DEFAULT_MODE)
            mode = DEFAULT_MODE
        self.mode = mode
        self.rrf_k = max(1, int(settings.rag_rrf_k or DEFAULT_RRF_K))
        self.prefixes = use_prefixes(settings.rag_prefixes, settings.rag_model)
        self.db_path = Path(settings.db_path).parent / "rag.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self.vector_backend = self._load_vector_backend()
        self.fts_enabled = False
        self._init_schema()
        self._sync_derived()
        logger.info("RAG: %s", self.describe())

    # ------------------------------------------------------------------ setup

    def _load_vector_backend(self) -> str:
        wanted = (self.settings.rag_vector_backend or "auto").lower()
        if wanted not in VECTOR_BACKENDS:
            logger.warning("JARVIS_RAG_VECTOR_BACKEND inválido (%r); usando 'auto'.", wanted)
            wanted = "auto"
        if wanted == "numpy":
            return "numpy"
        try:
            import sqlite_vec  # type: ignore

            self._db.enable_load_extension(True)
            try:
                sqlite_vec.load(self._db)
            finally:
                self._db.enable_load_extension(False)
            self._db.execute("SELECT vec_version()").fetchone()
        except Exception as exc:  # noqa: BLE001 - ImportError, AttributeError, sqlite3.Error
            logger.warning(
                "sqlite-vec indisponível (%s: %s); busca vetorial em NumPy.",
                type(exc).__name__,
                exc,
            )
            return "numpy"
        return "sqlite-vec"

    def describe(self) -> str:
        return "modo={}, vetorial={}, fts5={}, prefixos={}, rrf_k={}".format(
            self.mode,
            self.vector_backend,
            "sim" if self.fts_enabled else "não",
            "sim" if self.prefixes else "não",
            self.rrf_k,
        )

    @property
    def embed_sig(self) -> str:
        return "{}|{}".format(self.settings.rag_model, DOC_PREFIX if self.prefixes else "")

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
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(documents)")}
            if "embed_sig" not in columns:
                # Migração: índices antigos ganham a coluna vazia ("mesmo modelo, sem prefixo").
                self._db.execute(
                    "ALTER TABLE documents ADD COLUMN embed_sig TEXT NOT NULL DEFAULT ''"
                )
            try:
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING "
                    "fts5(text, tokenize='unicode61 remove_diacritics 2')"
                )
                self.fts_enabled = True
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "SQLite sem FTS5 (%s): modo bm25 vira vector e o híbrido usa só "
                    "a busca vetorial.",
                    exc,
                )
            self._db.commit()

    def _meta_get(self, key: str) -> Optional[str]:
        row = self._db.execute("SELECT value FROM rag_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO rag_meta (key, value) VALUES (?, ?)",
            (key, value),
        )

    def _vec_table_exists(self) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'"
        ).fetchone()
        return row is not None

    def _create_vec_table(self, dim: int) -> None:
        """(Re)cria chunks_vec vazia com a dimensão dada. Chamar com o lock."""
        if self._vec_table_exists():
            self._db.execute("DROP TABLE chunks_vec")
        self._db.execute(
            "CREATE VIRTUAL TABLE chunks_vec USING "
            "vec0(embedding float[{}] distance_metric=cosine)".format(int(dim))
        )
        self._meta_set("vec_dim", str(int(dim)))

    def _sync_derived(self) -> None:
        """Reconstrói chunks_fts/chunks_vec a partir de chunks se os ids divergirem."""
        with self._db_lock:
            chunk_ids = {row[0] for row in self._db.execute("SELECT id FROM chunks")}
            if self.fts_enabled:
                fts_ids = {row[0] for row in self._db.execute("SELECT rowid FROM chunks_fts")}
                if fts_ids != chunk_ids:
                    self._db.execute("DELETE FROM chunks_fts")
                    self._db.execute(
                        "INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks"
                    )
                    logger.info("Índice FTS5 reconstruído (%s trechos).", len(chunk_ids))
            if self.vector_backend == "sqlite-vec":
                vec_ids = set()
                if self._vec_table_exists():
                    vec_ids = {row[0] for row in self._db.execute("SELECT rowid FROM chunks_vec")}
                if vec_ids != chunk_ids:
                    self._rebuild_vec()
            self._db.commit()

    def _rebuild_vec(self) -> None:
        """Copia os embeddings JSON de chunks para chunks_vec. Chamar com o lock."""
        rows = self._db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        vectors: List[Tuple[int, List[float]]] = []
        for row in rows:
            try:
                vector = [float(v) for v in json.loads(row["embedding"])]
            except (ValueError, TypeError):
                continue
            if vector:
                vectors.append((int(row["id"]), vector))
        if not vectors:
            if self._vec_table_exists():
                self._db.execute("DELETE FROM chunks_vec")
            return
        # Dimensão do trecho mais recente (troca de modelo sem reindexar deixa os
        # antigos de fora, como a busca em NumPy já fazia).
        dim = len(vectors[-1][1])
        self._create_vec_table(dim)
        self._db.executemany(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            [(cid, _vec_blob(v)) for cid, v in vectors if len(v) == dim],
        )
        logger.info("Índice sqlite-vec reconstruído (%s trechos, dim %s).", len(vectors), dim)

    # ------------------------------------------------------------- embeddings

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

    def _embed_documents(self, chunks: Sequence[str]) -> List[List[float]]:
        texts = [DOC_PREFIX + chunk for chunk in chunks] if self.prefixes else list(chunks)
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            embeddings.extend(self._embed_batch(texts[start : start + EMBED_BATCH_SIZE]))
        return embeddings

    def _embed_query(self, query: str) -> List[float]:
        return self._embed(QUERY_PREFIX + query if self.prefixes else query)

    # ---------------------------------------------------------------- arquivos

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
                "SELECT id, mtime, size, embed_sig FROM documents WHERE path = ?", (path,)
            ).fetchone()

    def _delete_document_rows(self, document_id: int) -> None:
        """Apaga um documento, seus trechos e as linhas derivadas. Chamar com o lock."""
        chunk_ids = [
            (row[0],)
            for row in self._db.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            )
        ]
        if chunk_ids:
            if self.fts_enabled:
                self._db.executemany("DELETE FROM chunks_fts WHERE rowid = ?", chunk_ids)
            if self.vector_backend == "sqlite-vec" and self._vec_table_exists():
                self._db.executemany("DELETE FROM chunks_vec WHERE rowid = ?", chunk_ids)
        self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def _delete_document(self, path: str) -> None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT id FROM documents WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return
            self._delete_document_rows(int(row["id"]))
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
                self._delete_document_rows(int(row["id"]))
            cursor = self._db.execute(
                "INSERT INTO documents (path, mtime, size, indexed_at, embed_sig) "
                "VALUES (?, ?, ?, ?, ?)",
                (path, mtime, size, datetime.now().astimezone().isoformat(), self.embed_sig),
            )
            document_id = cursor.lastrowid
            use_vec = self.vector_backend == "sqlite-vec"
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cursor = self._db.execute(
                    "INSERT INTO chunks (document_id, chunk_index, text, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (document_id, index, chunk, json.dumps(list(embedding))),
                )
                chunk_id = cursor.lastrowid
                if self.fts_enabled:
                    self._db.execute(
                        "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk)
                    )
                if use_vec and embedding:
                    dim = len(embedding)
                    if not self._vec_table_exists() or self._meta_get("vec_dim") != str(dim):
                        # Primeiro trecho do índice ou troca de modelo: recria com a
                        # dimensão nova (os antigos saem da busca até reindexar).
                        self._create_vec_table(dim)
                    self._db.execute(
                        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                        (chunk_id, _vec_blob(embedding)),
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

    def _is_current(self, existing, stat) -> bool:
        if existing is None:
            return False
        stored_sig = existing["embed_sig"] or (self.settings.rag_model + "|")
        return (
            abs(existing["mtime"] - stat.st_mtime) < 0.001
            and existing["size"] == stat.st_size
            and stored_sig == self.embed_sig
        )

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
                    # Arquivo igual E mesma assinatura de embeddings (modelo + prefixo):
                    # trocar o modelo ou ligar os prefixos re-embeda no próximo índice.
                    if not force and self._is_current(self._existing(path_str), stat):
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
                    embeddings = self._embed_documents(chunks)
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

    # ------------------------------------------------------------------ busca

    def _vector_hits(self, query: str, depth: int) -> Optional[List[Tuple[int, float]]]:
        """(id do trecho, cosseno) em ordem decrescente; None se o embedding falhar."""
        try:
            query_vector = self._embed_query(query)
        except RuntimeError as exc:
            logger.warning("Busca sem embeddings: %s", exc)
            return None
        if self.vector_backend == "sqlite-vec":
            try:
                with self._db_lock:
                    if not self._vec_table_exists():
                        return []
                    if self._meta_get("vec_dim") != str(len(query_vector)):
                        logger.warning(
                            "Dimensão do embedding da consulta (%s) difere do índice (%s); "
                            "rode /reindex.",
                            len(query_vector),
                            self._meta_get("vec_dim"),
                        )
                        return []
                    rows = self._db.execute(
                        "SELECT rowid, distance FROM chunks_vec "
                        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                        (_vec_blob(query_vector), int(depth)),
                    ).fetchall()
                return [(int(row[0]), 1.0 - float(row[1])) for row in rows]
            except sqlite3.Error as exc:
                logger.warning("sqlite-vec falhou na busca (%s); usando NumPy.", exc)
        return self._numpy_hits(query_vector, depth)

    def _numpy_hits(self, query_vector: List[float], depth: int) -> List[Tuple[int, float]]:
        try:
            import numpy as np
        except ImportError:
            return []
        query = np.array(query_vector, dtype="float32")
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []
        with self._db_lock:
            rows = self._db.execute("SELECT id, embedding FROM chunks").fetchall()
        scored: List[Tuple[int, float]] = []
        for row in rows:
            try:
                vector = np.array(json.loads(row["embedding"]), dtype="float32")
            except (ValueError, TypeError):
                continue
            norm = float(np.linalg.norm(vector))
            if norm == 0.0 or vector.shape != query.shape:
                continue
            scored.append((int(row["id"]), float(np.dot(query, vector) / (query_norm * norm))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:depth]

    def _bm25_hits(self, query: str, depth: int) -> List[Tuple[int, float]]:
        """(id do trecho, -bm25) em ordem decrescente de relevância."""
        match = fts_query(query)
        if not match or not self.fts_enabled:
            return []
        try:
            with self._db_lock:
                rows = self._db.execute(
                    "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, int(depth)),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 falhou na busca (%s).", exc)
            return []
        return [(int(row[0]), -float(row[1])) for row in rows]

    def search(
        self, query: str, k: int = 5, mode: Optional[str] = None
    ) -> List[Dict[str, object]]:
        """Trechos mais relevantes. Cada item: path, name, chunk, chunks_total, score,
        text, mode (o modo que de fato respondeu)."""
        query = (query or "").strip()
        if not query:
            return []
        k = max(1, int(k))
        mode = (mode or self.mode).lower()
        if mode not in RAG_MODES:
            mode = "vector"
        if mode == "bm25" and not self.fts_enabled:
            mode = "vector"

        if mode == "bm25":
            hits = self._bm25_hits(query, k)
        elif mode == "vector":
            hits = self._vector_hits(query, k) or []
        else:
            depth = max(k, HYBRID_CANDIDATES)
            vector = self._vector_hits(query, depth)
            lexical = self._bm25_hits(query, depth) if self.fts_enabled else []
            if vector is None:
                logger.warning("Busca híbrida sem embeddings: usando só BM25.")
                hits, mode = lexical[:k], "bm25"
            elif not self.fts_enabled:
                hits, mode = vector[:k], "vector"
            else:
                fused = rrf_fuse(
                    [[cid for cid, _ in vector], [cid for cid, _ in lexical]], self.rrf_k
                )
                hits = fused[:k]
        return self._hydrate(hits, mode)

    def _hydrate(self, hits: Sequence[Tuple[int, float]], mode: str) -> List[Dict[str, object]]:
        if not hits:
            return []
        ids = [cid for cid, _ in hits]
        placeholders = ",".join("?" for _ in ids)
        with self._db_lock:
            rows = self._db.execute(
                "SELECT c.id, c.text, c.chunk_index, d.path, "
                "(SELECT COUNT(*) FROM chunks c2 WHERE c2.document_id = c.document_id) AS total "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.id IN ({})".format(placeholders),
                ids,
            ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        results: List[Dict[str, object]] = []
        for cid, score in hits:
            row = by_id.get(cid)
            if row is None:
                continue
            results.append(
                {
                    "path": row["path"],
                    "name": Path(row["path"]).name,
                    "chunk": int(row["chunk_index"]),
                    "chunks_total": int(row["total"]),
                    "score": float(score),
                    "text": row["text"],
                    "mode": mode,
                }
            )
        return results

    def stats(self) -> Dict[str, int]:
        with self._db_lock:
            documents = self._db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": int(documents), "chunks": int(chunks)}

    def close(self) -> None:
        with self._db_lock:
            self._db.close()
