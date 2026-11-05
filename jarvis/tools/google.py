from __future__ import annotations

import base64
import html
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import Settings
from .base import ToolRegistry


class GoogleTools:
    SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/gmail.modify",
    ]

    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _credentials(self) -> Tuple[Optional[Any], Optional[str]]:
        token_path = self.settings.google_token
        if not token_path.exists():
            return None, (
                "Google não configurado. Coloque o credentials.json no projeto e rode "
                "`python scripts/google_auth.py` uma vez."
            )
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
        except ImportError:
            return None, (
                "Dependências do Google não instaladas. Rode "
                "`pip install -r requirements.txt` no .venv."
            )
        try:
            credentials = Credentials.from_authorized_user_file(
                str(token_path), self.SCOPES
            )
            if not credentials.valid:
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                    token_path.write_text(credentials.to_json(), encoding="utf-8")
                else:
                    return None, (
                        "Token do Google inválido/expirado. "
                        "Rode `python scripts/google_auth.py` novamente."
                    )
        except Exception as exc:  # noqa: BLE001
            return None, "Erro ao carregar credenciais do Google: {}".format(exc)
        return credentials, None

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        credentials, err = self._credentials()
        if err:
            return None, err
        headers = {"Authorization": "Bearer " + (credentials.token or "")}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method, url, headers=headers, params=params, json=body
                )
        except httpx.HTTPError as exc:
            return None, "Erro de rede no Google: {}".format(exc)

        if response.status_code == 401:
            return None, (
                "Google recusou o token (401). Rode `python scripts/google_auth.py` "
                "para renovar."
            )
        if response.status_code == 403:
            return None, (
                "Google negou acesso (403). Verifique se as APIs Calendar e Gmail estão "
                "habilitadas e se os escopos foram autorizados."
            )
        if response.status_code >= 400:
            return None, "Google HTTP {}: {}".format(
                response.status_code, response.text[:600]
            )
        if response.status_code == 204 or not response.content:
            return {}, None
        try:
            return response.json(), None
        except ValueError:
            return None, "Resposta inválida do Google."

    @staticmethod
    def _normalize_datetime(value: str) -> str:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.isoformat()

    @staticmethod
    def _google_date(value: str) -> Dict[str, str]:
        text = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return {"date": text}
        return {"dateTime": GoogleTools._normalize_datetime(text)}

    @staticmethod
    def _decode_base64(data: str) -> str:
        try:
            padded = data + "=" * (-len(data) % 4)
            return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    @classmethod
    def _collect_bodies(cls, payload: Dict[str, Any], acc: Dict[str, List[str]]) -> None:
        mime = payload.get("mimeType") or ""
        data = (payload.get("body") or {}).get("data")
        if data and mime in {"text/plain", "text/html"}:
            decoded = cls._decode_base64(data)
            if decoded:
                acc.setdefault(mime, []).append(decoded)
        for part in payload.get("parts") or []:
            cls._collect_bodies(part, acc)

    @classmethod
    def _extract_body(cls, payload: Dict[str, Any]) -> str:
        acc: Dict[str, List[str]] = {}
        cls._collect_bodies(payload, acc)
        plain = "\n".join(acc.get("text/plain") or [])
        if plain.strip():
            return plain
        raw_html = "\n".join(acc.get("text/html") or [])
        text = re.sub(r"<[^>]+>", " ", raw_html)
        return html.unescape(" ".join(text.split()))

    @staticmethod
    def _headers(message: Dict[str, Any]) -> Dict[str, str]:
        payload = message.get("payload") or {}
        return {
            (header.get("name") or "").lower(): header.get("value") or ""
            for header in payload.get("headers") or []
        }

    def _google(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action", "")).strip().lower()
        try:
            limit = max(1, min(int(args.get("limit") or 10), 50))
        except (TypeError, ValueError):
            limit = 10

        if action == "calendar_list":
            try:
                days = max(1, min(int(args.get("days") or 7), 90))
            except (TypeError, ValueError):
                days = 7
            now = datetime.now().astimezone()
            data, err = self._request(
                "GET",
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                params={
                    "timeMin": now.isoformat(),
                    "timeMax": (now + timedelta(days=days)).isoformat(),
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": limit,
                },
            )
            if err:
                return err
            items = data.get("items") or []
            if not items:
                return "Nenhum evento nos próximos {} dia(s).".format(days)
            lines = []
            for event in items:
                start = event.get("start") or {}
                when = start.get("dateTime") or start.get("date") or "-"
                line = "- {} | {}".format(when, event.get("summary") or "(sem título)")
                if event.get("location"):
                    line += " @ " + event["location"]
                lines.append(line)
            return "\n".join(lines)

        if action == "calendar_create":
            summary = str(args.get("summary") or "").strip()
            start = str(args.get("start") or "").strip()
            end = str(args.get("end") or "").strip()
            if not summary or not start or not end:
                return "Informe 'summary', 'start' e 'end' (ISO 8601 ou AAAA-MM-DD)."
            body: Dict[str, Any] = {
                "summary": summary,
                "start": self._google_date(start),
                "end": self._google_date(end),
            }
            if args.get("description"):
                body["description"] = str(args["description"])
            if args.get("location"):
                body["location"] = str(args["location"])
            data, err = self._request(
                "POST",
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                body=body,
            )
            if err:
                return err
            return "Evento criado: {} ({})".format(
                data.get("summary") or summary, data.get("htmlLink") or data.get("id")
            )

        if action == "gmail_search":
            query = str(args.get("query") or "").strip()
            data, err = self._request(
                "GET",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                params={"q": query, "maxResults": limit},
            )
            if err:
                return err
            messages = data.get("messages") or []
            if not messages:
                return "Nenhuma mensagem encontrada para {!r}.".format(query)
            lines = []
            for item in messages:
                message_id = item.get("id")
                detail, detail_err = self._request(
                    "GET",
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/{}".format(
                        message_id
                    ),
                    params={
                        "format": "metadata",
                        "metadataHeaders": ["From", "Subject", "Date"],
                    },
                )
                if detail_err:
                    continue
                headers = self._headers(detail)
                lines.append(
                    "- {} | {} | {} [id: {}]".format(
                        headers.get("date", "-"),
                        headers.get("from", "-"),
                        headers.get("subject", "(sem assunto)"),
                        message_id,
                    )
                )
            return "\n".join(lines) or "Não consegui ler as mensagens."

        if action == "gmail_read":
            message_id = str(args.get("message_id") or "").strip()
            if not message_id:
                return "Informe 'message_id'."
            data, err = self._request(
                "GET",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/{}".format(
                    message_id
                ),
                params={"format": "full"},
            )
            if err:
                return err
            headers = self._headers(data)
            body_text = self._extract_body(data.get("payload") or {})
            return (
                "De: {}\nPara: {}\nData: {}\nAssunto: {}\n\n{}"
            ).format(
                headers.get("from", "-"),
                headers.get("to", "-"),
                headers.get("date", "-"),
                headers.get("subject", "(sem assunto)"),
                body_text[:8000] or "(corpo vazio)",
            )

        if action == "gmail_send":
            to = str(args.get("to") or "").strip()
            subject = str(args.get("subject") or "").strip()
            body_text = str(args.get("body") or "")
            if not to or not subject:
                return "Informe 'to' e 'subject'."
            try:
                from email.mime.text import MIMEText
            except ImportError:
                return "Não consegui importar o módulo de e-mail."
            message = MIMEText(body_text, "plain", "utf-8")
            message["To"] = to
            message["Subject"] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            data, err = self._request(
                "POST",
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                body={"raw": raw},
            )
            if err:
                return err
            return "E-mail enviado para {} (id: {}).".format(to, data.get("id"))

        return (
            "Ação desconhecida. Use: calendar_list, calendar_create, "
            "gmail_search, gmail_read, gmail_send."
        )

    def _register(self) -> None:
        self.registry.register(
            "google",
            "Acessa Google Agenda e Gmail: listar/criar eventos, buscar/ler/enviar e-mails. "
            "Requer credenciais OAuth (scripts/google_auth.py).",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "calendar_list",
                            "calendar_create",
                            "gmail_search",
                            "gmail_read",
                            "gmail_send",
                        ],
                    },
                    "days": {"type": "integer", "description": "Dias à frente (calendar_list)."},
                    "limit": {"type": "integer", "description": "Máximo de itens."},
                    "summary": {"type": "string", "description": "Título do evento."},
                    "start": {"type": "string", "description": "Início ISO 8601 (calendar_create)."},
                    "end": {"type": "string", "description": "Fim ISO 8601 (calendar_create)."},
                    "description": {"type": "string", "description": "Descrição do evento."},
                    "location": {"type": "string", "description": "Local do evento."},
                    "query": {"type": "string", "description": "Busca do Gmail (ex.: from:x is:unread)."},
                    "message_id": {"type": "string", "description": "ID da mensagem (gmail_read)."},
                    "to": {"type": "string", "description": "Destinatário (gmail_send)."},
                    "subject": {"type": "string", "description": "Assunto (gmail_send)."},
                    "body": {"type": "string", "description": "Corpo do e-mail (gmail_send)."},
                },
                "required": ["action"],
            },
            self._google,
        )
