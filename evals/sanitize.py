#!/usr/bin/env python3
"""Tira dados da máquina (home e nome do usuário) dos artefatos versionados.

Os evals rodam com a pasta de trabalho num diretório temporário, que no Windows fica
dentro da home (``C:\\Users\\<usuário>\\AppData\\Local\\Temp\\...``). O system prompt
mostra esse caminho e o modelo o copia nas respostas — às vezes com barras escapadas
duas ou três vezes, com o acento como ``\\u00e3`` ou com o nome truncado, sem acento
ou reinventado no fim. Este módulo troca:

- o nome do usuário (e variações que começam igual) por ``<USER>``;
- o nome curto 8.3 do Windows (``ABCDEF~1``) por ``<USER>``;
- qualquer ``X:\\Users\\<alguém>`` (qualquer separador e nível de escape), e
  ``/home/<usuário>`` ou ``/Users/<usuário>`` da máquina, por ``<HOME>``.

``common.save_json`` passa tudo por aqui antes de gravar. Para limpar arquivos já
gravados (ou só conferir, com ``--check``):

    .venv/Scripts/python.exe evals/sanitize.py evals/results
    .venv/Scripts/python.exe evals/sanitize.py --check evals/results
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Optional, Pattern

HOME_PLACEHOLDER = "<HOME>"
USER_PLACEHOLDER = "<USER>"

# Separador de caminho com qualquer nível de escape: \, \\, \\\\ ou /.
_SEP = r"(?:\\+|/)"
# Continuação do nome depois do prefixo: letras, dígitos, \u00e3 (escapado uma ou mais
# vezes) e letras não ASCII. Pega "MariaEduardaSo\\u00e1rezK" inteiro.
_NAME_TAIL = r"(?:[A-Za-z0-9]|\\+u[0-9a-fA-F]{4}|[^\x00-\x7f\s])*"


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def current_usernames() -> List[str]:
    """Nomes do usuário desta máquina (pasta da home e variáveis de ambiente)."""
    names = []
    candidates = [Path.home().name, os.environ.get("USERNAME"), os.environ.get("USER")]
    try:
        candidates.append(getpass.getuser())
    except Exception:  # noqa: BLE001 - sem usuário identificável
        pass
    for name in candidates:
        if name and name not in names and name.lower() not in {"root", "runner", "user"}:
            names.append(name)
    return names


def _char_pattern(char: str) -> str:
    """Um caractere do nome: literal, sem acento ou como escape \\uXXXX."""
    folded = _fold(char)
    if folded == char:
        return re.escape(char)
    options = [re.escape(char), r"\\+u{:04x}".format(ord(char))]
    if folded:
        options.append(re.escape(folded))
    return "(?:{})".format("|".join(options))


def user_pattern(username: str) -> Optional[Pattern[str]]:
    """Regex do nome do usuário e de variações que começam igual.

    Nomes com 6+ caracteres casam pelo prefixo (até 10 caracteres) seguido de qualquer
    continuação, porque o modelo trunca e reinventa o fim do nome. Nomes curtos só
    casam inteiros, entre fronteiras de palavra, para não trocar palavras comuns.
    """
    if not username:
        return None
    if len(username) >= 6:
        prefix = "".join(_char_pattern(char) for char in username[:10])
        body = prefix + _NAME_TAIL
    else:
        body = r"(?<![A-Za-z0-9])" + "".join(_char_pattern(c) for c in username) + r"(?![A-Za-z0-9])"
    return re.compile(body, re.IGNORECASE)


def short_name_pattern(username: str) -> Optional[Pattern[str]]:
    """Nome curto 8.3 do Windows: 6 primeiros caracteres sem acento + ~N."""
    folded = re.sub(r"[^A-Za-z0-9]", "", _fold(username)).upper()
    if len(folded) < 3:
        return None
    return re.compile(re.escape(folded[:6]) + r"~\d+", re.IGNORECASE)


class Sanitizer:
    def __init__(self, usernames: Optional[Iterable[str]] = None) -> None:
        names = list(usernames) if usernames is not None else current_usernames()
        self._user_patterns: List[Pattern[str]] = []
        for name in names:
            for pattern in (user_pattern(name), short_name_pattern(name)):
                if pattern is not None:
                    self._user_patterns.append(pattern)
        user = re.escape(USER_PLACEHOLDER)
        self._home_patterns = [
            # C:\Users\<qualquer um> com qualquer separador/escape
            re.compile(r"[A-Za-z]:" + _SEP + r"Users" + _SEP + r"(?:" + user + r"|[^\\/\"'\s:*?<>|]+)",
                       re.IGNORECASE),
            # /home/<usuário> e /Users/<usuário> só quando é o usuário desta máquina
            re.compile(r"(?<![A-Za-z0-9])/(?:home|Users)/" + user),
        ]

    def text(self, value: str) -> str:
        for pattern in self._user_patterns:
            value = pattern.sub(USER_PLACEHOLDER, value)
        for pattern in self._home_patterns:
            value = pattern.sub(HOME_PLACEHOLDER, value)
        return value

    def obj(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(k) if isinstance(k, str) else k: self.obj(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.obj(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.obj(item) for item in value)
        return value


_DEFAULT: Optional[Sanitizer] = None


def default() -> Sanitizer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Sanitizer()
    return _DEFAULT


def sanitize_text(value: str) -> str:
    return default().text(value)


def sanitize_obj(value: Any) -> Any:
    return default().obj(value)


# ---------------------------------------------------------------------- CLI


def _dump_json(payload: Any) -> str:
    # mesmo formato de common.save_json
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def sanitize_file(path: Path, sanitizer: Sanitizer, write: bool = True) -> bool:
    """Devolve True se o arquivo tinha algo a trocar (e o regrava se write=True)."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        clean = _dump_json(sanitizer.obj(json.loads(raw)))
        changed = clean != _dump_json(json.loads(raw))
    else:
        clean = sanitizer.text(raw)
        changed = clean != raw
    if changed and write:
        with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(clean)
    return changed


def _iter_files(targets: Iterable[str]) -> Iterable[Path]:
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for item in sorted(path.rglob("*")):
                if item.suffix in {".json", ".md", ".jsonl", ".txt"} and item.is_file():
                    yield item
        elif path.is_file():
            yield path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", default=["evals/results"])
    parser.add_argument("--check", action="store_true", help="só confere; sai com 1 se achar algo")
    parser.add_argument("--user", action="append", help="nome de usuário extra a remover")
    args = parser.parse_args(argv)

    names = current_usernames() + (args.user or [])
    sanitizer = Sanitizer(names)
    found = [p for p in _iter_files(args.paths) if sanitize_file(p, sanitizer, write=not args.check)]
    verb = "com dados da máquina" if args.check else "limpos"
    print("{} arquivo(s) {}".format(len(found), verb))
    for path in found:
        print("  " + path.as_posix())
    return 1 if (args.check and found) else 0


if __name__ == "__main__":
    sys.exit(main())
