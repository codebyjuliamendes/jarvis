from __future__ import annotations

import fnmatch
import ipaddress
import re
import socket
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse


class SecurityError(Exception):
    """Erro de política de segurança (caminho/comando bloqueado)."""


SENSITIVE_DIRS = {
    ".ssh",
    ".gnupg",
    ".pki",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".config",
    ".claude",
    ".cursor",
    ".gemini",
    ".copilot",
    ".git",
}

SENSITIVE_GLOBS = [
    ".env",
    ".env.*",
    "*.env",
    "credentials.json",
    "*credential*.json",
    "client_secret*.json",
    "token.json",
    "ms_graph_token.json",
    "*graph_token*.json",
    "id_rsa*",
    "id_ed25519*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    ".netrc",
    ".pgpass",
    ".bash_history",
    ".python_history",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".htpasswd",
]

_DANGEROUS_COMMANDS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bsudo\b",
        r"\bsu\s+-",
        r"\bmkfs(?:\.\w+)?\b",
        r"\bdd\s+if=",
        r"\b(?:shutdown|reboot|poweroff|halt)\b",
        r"\binit\s+[06]\b",
        r"\bsystemctl\s+(?:poweroff|reboot|halt)\b",
        r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(?:/|/\*|~|~/|\$HOME)\s*$",
        r":\s*\(\s*\)\s*\{.*\|.*&.*\}",
        r">\s*/dev/(?:sd|nvme|hd)",
        r"\bchmod\s+-R\s+777\s+/\s*$",
        r"\b(?:passwd|visudo)\b",
        r"\b(?:curl|wget)\b[^|]*\|\s*(?:ba|z|k|da)?sh\b",
    ]
]


def is_sensitive(path: Path) -> bool:
    """True se o caminho aponta para credenciais/segredos."""
    parts = [part.lower() for part in path.parts]
    if any(part in SENSITIVE_DIRS for part in parts):
        return True
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_GLOBS)


def resolve_path(
    raw: str,
    roots: Iterable[Path],
    base: Optional[Path] = None,
    must_exist: bool = False,
    allow_sensitive: bool = False,
) -> Path:
    """Resolve e valida um caminho contra as pastas permitidas."""
    if raw is None or str(raw).strip() == "":
        raise SecurityError("caminho vazio")

    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    path = path.resolve(strict=False)

    if must_exist and not path.exists():
        raise SecurityError("caminho não existe: {}".format(path))

    if not allow_sensitive and is_sensitive(path):
        raise SecurityError(
            "caminho bloqueado por segurança (credenciais/segredos): {}".format(path)
        )

    for root in roots:
        try:
            path.relative_to(Path(root).resolve(strict=False))
            return path
        except ValueError:
            continue

    allowed = ", ".join(str(Path(r).resolve(strict=False)) for r in roots)
    raise SecurityError("caminho fora das pastas permitidas ({}): {}".format(allowed, path))


def check_shell_command(command: str) -> None:
    """Bloqueia comandos claramente destrutivos. Não é um sandbox completo."""
    if not command or not command.strip():
        raise SecurityError("comando vazio")

    for pattern in _DANGEROUS_COMMANDS:
        if pattern.search(command):
            raise SecurityError(
                "comando bloqueado por segurança (padrão proibido: {})".format(
                    pattern.pattern
                )
            )


def check_url(url: str) -> str:
    """Valida uma URL para web_fetch, bloqueando alvos internos (SSRF básico)."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise SecurityError("apenas URLs http/https são permitidas")
    if not parsed.hostname:
        raise SecurityError("URL sem host")
    if parsed.username or parsed.password:
        raise SecurityError("URLs com usuário/senha embutidos são bloqueadas")

    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise SecurityError("acesso a hosts locais é bloqueado")

    addresses = []
    try:
        addresses.append(ipaddress.ip_address(host))
    except ValueError:
        try:
            for info in socket.getaddrinfo(host, None):
                addresses.append(ipaddress.ip_address(info[4][0]))
        except (socket.gaierror, ValueError):
            addresses = []

    for address in addresses:
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise SecurityError("acesso a endereços internos é bloqueado")

    return parsed.geturl()
