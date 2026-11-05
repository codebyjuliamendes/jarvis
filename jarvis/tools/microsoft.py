from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
import pytz

from ..config import Settings
from ..graph_auth import get_access_token
from .base import ToolRegistry

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return html.unescape(" ".join(text.split()))


def _quote(value: str) -> str:
    return quote(str(value), safe="")


class MicrosoftTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        token, err = get_access_token(self.settings)
        if err:
            return None, err
        request_headers = {"Authorization": "Bearer " + (token or "")}
        if headers:
            request_headers.update(headers)
        url = path if path.startswith("http") else GRAPH_BASE + path
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method,
                    url,
                    headers=request_headers,
                    params=params,
                    json=body,
                )
        except httpx.HTTPError as exc:
            return None, "Erro de rede na Microsoft: {}".format(exc)

        if response.status_code == 401:
            return None, (
                "Microsoft recusou o token (401). Rode "
                "`python scripts/ms_graph_auth.py` novamente."
            )
        if response.status_code == 403:
            return None, (
                "Microsoft negou acesso (403). Pode faltar permissão ou consentimento "
                "do admin no tenant. Detalhe: {}".format(response.text[:400])
            )
        if response.status_code >= 400:
            return None, "Microsoft HTTP {}: {}".format(
                response.status_code, response.text[:600]
            )
        if response.status_code == 204 or not response.content:
            return {}, None
        try:
            return response.json(), None
        except ValueError:
            return None, "Resposta inválida da Microsoft."

    def _localize(self, value: str) -> datetime:
        text = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            text += "T09:00:00"
        text = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = pytz.timezone(self.settings.timezone).localize(parsed)
        return parsed

    def _graph_datetime(self, parsed: datetime) -> Dict[str, str]:
        return {
            "dateTime": parsed.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": self.settings.timezone,
        }

    @staticmethod
    def _short_dt(value: Any) -> str:
        text = str(value or "")
        if "T" in text:
            text = text.replace("T", " ")
        return text[:16] if len(text) >= 16 else text

    @classmethod
    def _format_event(cls, event: Dict[str, Any]) -> str:
        start = event.get("start") or {}
        end = event.get("end") or {}
        when = cls._short_dt(start.get("dateTime") or start.get("date"))
        end_when = cls._short_dt(end.get("dateTime") or end.get("date"))
        if end_when and end_when != when:
            when = "{} -> {}".format(when, end_when)
        line = "- {} | {}".format(
            when or "-",
            event.get("subject") or event.get("summary") or "(sem título)",
        )
        location = (event.get("location") or {}).get("displayName")
        if location:
            line += " @ " + location
        if event.get("id"):
            line += " [id: {}]".format(event["id"])
        return line

    def _microsoft(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action") or "").strip().lower()
        try:
            limit = max(1, min(int(args.get("limit") or 10), 50))
        except (TypeError, ValueError):
            limit = 10

        if action == "me":
            data, err = self._request(
                "GET", "/me", params={"$select": "displayName,mail,userPrincipalName"}
            )
            if err:
                return err
            return "Microsoft: {} <{}>".format(
                data.get("displayName") or "-",
                data.get("mail") or data.get("userPrincipalName") or "-",
            )

        if action == "calendar_list":
            try:
                days = max(1, min(int(args.get("days") or 7), 90))
            except (TypeError, ValueError):
                days = 7
            timezone = pytz.timezone(self.settings.timezone)
            now = datetime.now(timezone)
            data, err = self._request(
                "GET",
                "/me/calendarView",
                params={
                    "startDateTime": now.isoformat(),
                    "endDateTime": (now + timedelta(days=days)).isoformat(),
                    "$top": limit,
                    "$select": "id,subject,start,end,location,organizer,webLink",
                    "$orderby": "start/dateTime",
                },
                headers={"Prefer": 'outlook.timezone="{}"'.format(self.settings.timezone)},
            )
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhum evento nos próximos {} dia(s).".format(days)
            return "\n".join(self._format_event(event) for event in items)

        if action == "calendar_create":
            summary = str(args.get("summary") or "").strip()
            start_raw = str(args.get("start") or "").strip()
            end_raw = str(args.get("end") or "").strip()
            if not summary or not start_raw or not end_raw:
                return "Informe 'summary', 'start' e 'end' (ISO 8601 ou AAAA-MM-DD)."
            try:
                start = self._localize(start_raw)
                end = self._localize(end_raw)
            except ValueError:
                return "Data inválida. Use ISO 8601, ex.: 2026-11-20T14:00:00."
            body: Dict[str, Any] = {
                "subject": summary,
                "start": self._graph_datetime(start),
                "end": self._graph_datetime(end),
            }
            if args.get("description"):
                body["body"] = {
                    "contentType": "Text",
                    "content": str(args["description"]),
                }
            if args.get("location"):
                body["location"] = {"displayName": str(args["location"])}
            attendees = [
                item.strip()
                for item in re.split(r"[,;]", str(args.get("attendees") or ""))
                if item.strip()
            ]
            if attendees:
                body["attendees"] = [
                    {"emailAddress": {"address": address}, "type": "required"}
                    for address in attendees
                ]
            data, err = self._request("POST", "/me/events", body=body)
            if err:
                return err
            return "Evento criado: {} ({})".format(
                data.get("subject") or summary,
                data.get("webLink") or data.get("id"),
            )

        if action == "calendar_delete":
            event_id = str(args.get("event_id") or "").strip()
            if not event_id:
                return "Informe 'event_id'."
            _, err = self._request(
                "DELETE", "/me/events/{}".format(_quote(event_id))
            )
            if err:
                return err
            return "Evento removido."

        if action == "mail_list":
            unread = bool(args.get("unread", True))
            params = {
                "$top": limit,
                "$select": "id,subject,from,receivedDateTime,isRead,webLink,bodyPreview",
                "$orderby": "receivedDateTime desc",
            }
            if unread:
                params["$filter"] = "isRead eq false"
            data, err = self._request("GET", "/me/messages", params=params)
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhum e-mail encontrado."
            lines = []
            for message in items:
                sender = ((message.get("from") or {}).get("emailAddress") or {})
                lines.append(
                    "- {} | {} | {} [id: {}]".format(
                        message.get("receivedDateTime") or "-",
                        sender.get("address") or "-",
                        message.get("subject") or "(sem assunto)",
                        message.get("id") or "-",
                    )
                )
            return "\n".join(lines)

        if action == "mail_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return "Informe 'query'."
            data, err = self._request(
                "GET",
                "/me/messages",
                params={
                    "$search": '"{}"'.format(query.replace('"', " ")),
                    "$top": limit,
                    "$select": "id,subject,from,receivedDateTime,isRead,webLink",
                },
            )
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhum e-mail encontrado para {!r}.".format(query)
            lines = []
            for message in items:
                sender = ((message.get("from") or {}).get("emailAddress") or {})
                lines.append(
                    "- {} | {} | {} [id: {}]".format(
                        message.get("receivedDateTime") or "-",
                        sender.get("address") or "-",
                        message.get("subject") or "(sem assunto)",
                        message.get("id") or "-",
                    )
                )
            return "\n".join(lines)

        if action == "mail_read":
            message_id = str(args.get("message_id") or "").strip()
            if not message_id:
                return "Informe 'message_id'."
            data, err = self._request(
                "GET",
                "/me/messages/{}".format(_quote(message_id)),
                params={
                    "$select": "subject,from,toRecipients,receivedDateTime,body,webLink"
                },
            )
            if err:
                return err
            sender = ((data.get("from") or {}).get("emailAddress") or {}).get(
                "address"
            ) or "-"
            recipients = ", ".join(
                ((item.get("emailAddress") or {}).get("address") or "")
                for item in data.get("toRecipients") or []
            )
            content = _strip_html((data.get("body") or {}).get("content") or "")
            return (
                "De: {}\nPara: {}\nData: {}\nAssunto: {}\n\n{}"
            ).format(
                sender,
                recipients or "-",
                data.get("receivedDateTime") or "-",
                data.get("subject") or "(sem assunto)",
                content[:8000] or "(corpo vazio)",
            )

        if action in {"mail_mark_read", "mail_mark_unread"}:
            message_id = str(args.get("message_id") or "").strip()
            if not message_id:
                return "Informe 'message_id'."
            read = action == "mail_mark_read"
            _, err = self._request(
                "PATCH",
                "/me/messages/{}".format(_quote(message_id)),
                body={"isRead": read},
            )
            if err:
                return err
            return "E-mail marcado como {}.".format("lido" if read else "não lido")

        if action == "mail_send":
            to = str(args.get("to") or "").strip()
            subject = str(args.get("subject") or "").strip()
            body_text = str(args.get("body") or "")
            if not to or not subject:
                return "Informe 'to' e 'subject'."
            addresses = [
                item.strip()
                for item in re.split(r"[,;]", to)
                if item.strip() and "@" in item
            ]
            if not addresses:
                return "Endereço de e-mail inválido."
            payload = {
                "message": {
                    "subject": subject,
                    "body": {"contentType": "Text", "content": body_text},
                    "toRecipients": [
                        {"emailAddress": {"address": address}}
                        for address in addresses
                    ],
                },
                "saveToSentItems": True,
            }
            _, err = self._request("POST", "/me/sendMail", body=payload)
            if err:
                return err
            return "E-mail enviado para {}.".format(", ".join(addresses))

        if action == "teams_list":
            data, err = self._request(
                "GET",
                "/me/joinedTeams",
                params={"$select": "id,displayName,description"},
            )
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhuma equipe encontrada."
            return "\n".join(
                "- {} | {} [id: {}]".format(
                    team.get("displayName") or "-",
                    team.get("description") or "",
                    team.get("id") or "-",
                )
                for team in items
            )

        if action == "teams_channels":
            team_id = str(args.get("team_id") or "").strip()
            if not team_id:
                return "Informe 'team_id'."
            data, err = self._request(
                "GET",
                "/teams/{}/channels".format(_quote(team_id)),
                params={"$select": "id,displayName,description,webUrl"},
            )
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhum canal encontrado."
            return "\n".join(
                "- {} | {} [id: {}]".format(
                    channel.get("displayName") or "-",
                    channel.get("description") or "",
                    channel.get("id") or "-",
                )
                for channel in items
            )

        if action == "teams_chats":
            data, err = self._request(
                "GET",
                "/me/chats",
                params={
                    "$top": limit,
                    "$expand": "members",
                    "$select": "id,topic,chatType,lastUpdatedDateTime",
                },
            )
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhum chat encontrado."
            lines = []
            for chat in items:
                members = ", ".join(
                    (member.get("displayName") or "")
                    for member in chat.get("members") or []
                    if member.get("displayName")
                )
                lines.append(
                    "- {} | {} | {} [id: {}]".format(
                        chat.get("lastUpdatedDateTime") or "-",
                        chat.get("topic") or chat.get("chatType") or "-",
                        members or "-",
                        chat.get("id") or "-",
                    )
                )
            return "\n".join(lines)

        if action == "teams_messages":
            chat_id = str(args.get("chat_id") or "").strip()
            if not chat_id:
                return "Informe 'chat_id'."
            data, err = self._request(
                "GET",
                "/me/chats/{}/messages".format(_quote(chat_id)),
                params={"$top": limit},
            )
            if err:
                return err
            items = data.get("value") or []
            if not items:
                return "Nenhuma mensagem encontrada."
            lines = []
            for message in items:
                sender = ((message.get("from") or {}).get("user") or {}).get(
                    "displayName"
                ) or "-"
                content = _strip_html(
                    (message.get("body") or {}).get("content") or ""
                )
                lines.append(
                    "- {} | {}: {}".format(
                        message.get("createdDateTime") or "-",
                        sender,
                        content[:500] or "(sem texto)",
                    )
                )
            return "\n".join(lines)

        if action == "teams_channel_messages":
            team_id = str(args.get("team_id") or "").strip()
            channel_id = str(args.get("channel_id") or "").strip()
            if not team_id or not channel_id:
                return "Informe 'team_id' e 'channel_id'."
            data, err = self._request(
                "GET",
                "/teams/{}/channels/{}/messages".format(
                    _quote(team_id), _quote(channel_id)
                ),
                params={"$top": limit},
            )
            if err:
                return err + (
                    "\nLeitura de mensagens de canal geralmente exige "
                    "ChannelMessage.Read.All com consentimento do admin."
                )
            items = data.get("value") or []
            if not items:
                return "Nenhuma mensagem encontrada."
            lines = []
            for message in items:
                sender = ((message.get("from") or {}).get("user") or {}).get(
                    "displayName"
                ) or "-"
                content = _strip_html(
                    (message.get("body") or {}).get("content") or ""
                )
                lines.append(
                    "- {} | {}: {}".format(
                        message.get("createdDateTime") or "-",
                        sender,
                        content[:500] or "(sem texto)",
                    )
                )
            return "\n".join(lines)

        return (
            "Ação desconhecida. Use: me, calendar_list, calendar_create, calendar_delete, "
            "mail_list, mail_search, mail_read, mail_mark_read, mail_mark_unread, "
            "mail_send, teams_list, teams_channels, teams_chats, teams_messages, "
            "teams_channel_messages."
        )

    def _register(self) -> None:
        self.registry.register(
            "microsoft",
            "Acessa Microsoft 365 via Graph: perfil, Outlook (listar/buscar/ler/marcar/"
            "enviar e-mails), calendário (listar/criar/remover eventos) e Teams (equipes, "
            "canais, chats e mensagens). Requer MS_GRAPH_CLIENT_ID e "
            "scripts/ms_graph_auth.py.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "me",
                            "calendar_list",
                            "calendar_create",
                            "calendar_delete",
                            "mail_list",
                            "mail_search",
                            "mail_read",
                            "mail_mark_read",
                            "mail_mark_unread",
                            "mail_send",
                            "teams_list",
                            "teams_channels",
                            "teams_chats",
                            "teams_messages",
                            "teams_channel_messages",
                        ],
                    },
                    "days": {"type": "integer", "description": "Dias à frente (calendar_list)."},
                    "limit": {"type": "integer", "description": "Máximo de itens."},
                    "summary": {"type": "string", "description": "Título do evento."},
                    "start": {"type": "string", "description": "Início ISO 8601 (calendar_create)."},
                    "end": {"type": "string", "description": "Fim ISO 8601 (calendar_create)."},
                    "event_id": {"type": "string", "description": "ID do evento (calendar_delete)."},
                    "description": {"type": "string", "description": "Descrição do evento."},
                    "location": {"type": "string", "description": "Local do evento."},
                    "attendees": {
                        "type": "string",
                        "description": "Participantes separados por vírgula.",
                    },
                    "unread": {
                        "type": "boolean",
                        "description": "Só não lidos (mail_list, padrão true).",
                    },
                    "query": {"type": "string", "description": "Busca de e-mail."},
                    "message_id": {"type": "string", "description": "ID da mensagem (mail_read)."},
                    "to": {"type": "string", "description": "Destinatário(s) (mail_send)."},
                    "subject": {"type": "string", "description": "Assunto (mail_send)."},
                    "body": {"type": "string", "description": "Corpo do e-mail (mail_send)."},
                    "team_id": {"type": "string", "description": "ID da equipe (Teams)."},
                    "channel_id": {"type": "string", "description": "ID do canal (Teams)."},
                    "chat_id": {"type": "string", "description": "ID do chat (Teams)."},
                },
                "required": ["action"],
            },
            self._microsoft,
        )
