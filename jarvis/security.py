from __future__ import annotations

import fnmatch
import ipaddress
import posixpath
import re
import shlex
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
        r":\s*\(\s*\)\s*\{.*\|.*&.*\}",
        r">\s*/dev/(?:sd|nvme|hd)",
        r"\bchmod\s+-R\s+777\s+/\s*$",
        r"\b(?:passwd|visudo)\b",
        r"\b(?:curl|wget)\b[^|]*\|\s*(?:ba|z|k|da)?sh\b",
    ]
]


# ------------------------------------------------------------ rm recursivo
#
# Um regex só não dá conta das formas de pedir "rm recursivo": -rf, -fr, -r -f, -R,
# --recursive --force, abreviações longas (--rec), /bin/rm, \rm, env/nohup na frente,
# sh -c "...". Por isso o comando é dividido em segmentos, tokenizado com shlex e as
# flags são juntadas antes de olhar os alvos.

# Separadores de comandos: ; && || | & ( ) ` e quebra de linha (cobre $(...) também).
_SEGMENT_SPLIT = re.compile(r"\|\||&&|[;|&()`\n]")
# Prefixos que executam o comando seguinte (as opções deles são puladas).
_WRAPPERS = {
    "env", "nice", "nohup", "time", "command", "builtin", "exec", "busybox",
    "xargs", "timeout", "stdbuf", "ionice", "chrt", "doas",
}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Diretórios de primeiro nível cujo apagamento recursivo inutiliza o sistema.
_CRITICAL_DIRS = {
    "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64", "/media",
    "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
    "/Applications", "/Library", "/System", "/Users", "/Volumes",
}
_HOME_TARGETS = {"~", "$HOME", "${HOME}"}


def _split_tokens(segment: str) -> List[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:  # aspas sem fechar: cai para espaços
        return segment.split()


def _strip_wrappers(tokens: List[str]) -> List[str]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = posixpath.basename(token.lstrip("\\")).lower()
        if _ASSIGNMENT.match(token):
            index += 1
        elif name in _WRAPPERS:
            index += 1
            # opções do prefixo (env -i, nice -n 10, timeout 5, xargs -0...)
            while index < len(tokens) and (
                tokens[index].startswith("-") or tokens[index].isdigit()
                or _ASSIGNMENT.match(tokens[index])
            ):
                index += 1
        else:
            break
    return tokens[index:]


def _is_critical_target(raw: str) -> bool:
    target = raw.strip().strip("'\"")
    if not target:
        return False
    if re.match(r"^[A-Za-z]:[\\/]*\*?$", target):  # C:\ ou C:/ (Git Bash/WSL)
        return True
    path = target.replace("\\", "/")
    # tira barras repetidas e sufixos "/", "/.", "/*" ("/*" = tudo dentro)
    while True:
        previous = path
        path = re.sub(r"/{2,}", "/", path)
        if path.endswith("/*") or path.endswith("/."):
            path = path[:-2] or "/"
        elif len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        if path == previous:
            break
    if path.startswith("/"):
        path = posixpath.normpath(path)
        if path.startswith("//"):
            path = path[1:]
    if path == "/" or path in _HOME_TARGETS or path in _CRITICAL_DIRS:
        return True
    # ~outro_usuario (a home de outra pessoa)
    return re.match(r"^~[A-Za-z0-9._-]+$", path) is not None


def _dangerous_rm(tokens: List[str], depth: int = 0) -> Optional[str]:
    """Devolve o alvo crítico se os tokens forem um rm recursivo nele; senão None."""
    tokens = _strip_wrappers(tokens)
    if not tokens:
        return None
    program = posixpath.basename(tokens[0].lstrip("\\")).lower()
    if program in _SHELLS and depth < 3:
        # sh -c "rm -rf /": verifica o script passado ao shell
        for position, token in enumerate(tokens[1:], start=1):
            if token.startswith("-") and not token.startswith("--") and "c" in token:
                if position + 1 < len(tokens):
                    return _find_dangerous_rm(tokens[position + 1], depth + 1)
        return None
    if program != "rm":
        return None

    recursive = False
    targets: List[str] = []
    options_done = False
    for token in tokens[1:]:
        if options_done:
            targets.append(token)
        elif token == "--":
            options_done = True
        elif token.startswith("--"):
            name = token[2:].split("=", 1)[0].lower()
            # o GNU rm aceita abreviação única: --rec, --recursive
            if name and "recursive".startswith(name):
                recursive = True
        elif token.startswith("-") and len(token) > 1:
            if "r" in token[1:] or "R" in token[1:]:
                recursive = True
        else:
            targets.append(token)
    if not recursive:
        return None
    for target in targets:
        if _is_critical_target(target):
            return target
    return None


def _find_dangerous_rm(command: str, depth: int = 0) -> Optional[str]:
    for segment in _SEGMENT_SPLIT.split(command):
        target = _dangerous_rm(_split_tokens(segment), depth)
        if target is not None:
            return target
    return None


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

    target = _find_dangerous_rm(command)
    if target is not None:
        raise SecurityError(
            "comando bloqueado por segurança (rm recursivo em alvo crítico: {})".format(target)
        )

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
