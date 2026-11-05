from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import Settings
from .base import ToolRegistry


class BusinessTools:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._register()

    def _stripe(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        if not self.settings.stripe_api_key:
            return None, (
                "Stripe não configurado. Defina STRIPE_API_KEY no .env "
                "(prefira uma chave restrita rk_ ou de teste sk_test_)."
            )
        headers = {"Authorization": "Bearer " + self.settings.stripe_api_key}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method,
                    "https://api.stripe.com/v1" + path,
                    headers=headers,
                    params=params,
                )
        except httpx.HTTPError as exc:
            return None, "Erro de rede no Stripe: {}".format(exc)

        if response.status_code == 401:
            return None, "Stripe recusou a chave (401). Verifique STRIPE_API_KEY."
        if response.status_code >= 400:
            return None, "Stripe HTTP {}: {}".format(
                response.status_code, response.text[:600]
            )
        try:
            return response.json(), None
        except ValueError:
            return None, "Resposta inválida do Stripe."

    def _key_warning(self) -> str:
        key = self.settings.stripe_api_key or ""
        if key.startswith("sk_live_") or key.startswith("rk_live_"):
            return "\n[atenção] Você está usando uma chave LIVE do Stripe."
        return ""

    @staticmethod
    def _money(amount_cents: Any, currency: str) -> str:
        try:
            amount = float(amount_cents or 0) / 100.0
        except (TypeError, ValueError):
            amount = 0.0
        code = (currency or "").upper()
        symbol = {"BRL": "R$", "USD": "US$", "EUR": "€", "GBP": "£"}.get(code, code + " ")
        formatted = "{:,.2f}".format(amount)
        formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        return "{} {}".format(symbol, formatted)

    @staticmethod
    def _date(timestamp: Any) -> str:
        try:
            return datetime.fromtimestamp(int(timestamp)).strftime("%d/%m/%Y")
        except (TypeError, ValueError, OSError):
            return "-"

    def _stripe_tool(self, args: Dict[str, Any]) -> str:
        action = str(args.get("action", "")).strip().lower()
        try:
            limit = max(1, min(int(args.get("limit") or 10), 100))
        except (TypeError, ValueError):
            limit = 10

        if action == "overview":
            balance, err = self._stripe("GET", "/balance")
            if err:
                return err
            charges, charges_err = self._stripe(
                "GET", "/charges", params={"limit": min(limit, 10)}
            )
            lines: List[str] = ["Saldo Stripe:"]
            for kind in ("available", "pending"):
                for item in balance.get(kind) or []:
                    lines.append(
                        "- {}: {}".format(
                            "disponível" if kind == "available" else "pendente",
                            self._money(item.get("amount"), item.get("currency", "")),
                        )
                    )
            lines.append("")
            lines.append("Últimas cobranças:")
            if charges_err:
                lines.append("(não consegui listar: {})".format(charges_err))
            else:
                for charge in charges.get("data") or []:
                    lines.append(
                        "- {} | {} | {} | {}".format(
                            self._date(charge.get("created")),
                            self._money(charge.get("amount"), charge.get("currency", "")),
                            charge.get("status", "-"),
                            charge.get("description")
                            or (charge.get("billing_details") or {}).get("name")
                            or "(sem descrição)",
                        )
                    )
            return "\n".join(lines) + self._key_warning()

        if action == "revenue":
            try:
                days = max(1, min(int(args.get("days") or 30), 365))
            except (TypeError, ValueError):
                days = 30
            since = int(time.time()) - days * 86400
            data, err = self._stripe(
                "GET",
                "/charges",
                params={"created[gte]": since, "limit": 100},
            )
            if err:
                return err
            charges = data.get("data") or []
            successful = [c for c in charges if c.get("status") == "succeeded"]
            currency = (successful[0].get("currency") if successful else "") or ""
            gross = sum(int(c.get("amount") or 0) for c in successful)
            refunded = sum(int(c.get("amount_refunded") or 0) for c in successful)
            lines = [
                "Período: últimos {} dia(s).".format(days),
                "Cobranças bem-sucedidas: {}".format(len(successful)),
                "Receita bruta: {}".format(self._money(gross, currency)),
                "Reembolsos: {}".format(self._money(refunded, currency)),
                "Receita líquida: {}".format(self._money(gross - refunded, currency)),
            ]
            if data.get("has_more"):
                lines.append(
                    "[aviso] Havia mais de 100 cobranças no período; o total está parcial."
                )
            return "\n".join(lines) + self._key_warning()

        if action == "customers":
            params: Dict[str, Any] = {"limit": limit}
            email = str(args.get("email") or "").strip()
            if email:
                params["email"] = email
            data, err = self._stripe("GET", "/customers", params=params)
            if err:
                return err
            lines = []
            for customer in data.get("data") or []:
                lines.append(
                    "- {} | {} | {} | criado em {}".format(
                        customer.get("name") or "(sem nome)",
                        customer.get("email") or "(sem e-mail)",
                        customer.get("id"),
                        self._date(customer.get("created")),
                    )
                )
            return "\n".join(lines) or "Nenhum cliente encontrado."

        if action == "invoices":
            params = {"limit": limit}
            status = str(args.get("status") or "").strip()
            if status:
                params["status"] = status
            data, err = self._stripe("GET", "/invoices", params=params)
            if err:
                return err
            lines = []
            for invoice in data.get("data") or []:
                lines.append(
                    "- #{} | {} | {} | {} | {}".format(
                        invoice.get("number") or invoice.get("id"),
                        invoice.get("status", "-"),
                        self._money(
                            invoice.get("total"), invoice.get("currency", "")
                        ),
                        invoice.get("customer_email") or "-",
                        self._date(invoice.get("created")),
                    )
                )
            return "\n".join(lines) or "Nenhuma fatura encontrada."

        if action == "payment_links":
            data, err = self._stripe(
                "GET", "/payment_links", params={"limit": limit}
            )
            if err:
                return err
            lines = []
            for link in data.get("data") or []:
                lines.append(
                    "- {} | ativo: {} | {} | {}".format(
                        link.get("id"),
                        "sim" if link.get("active") else "não",
                        link.get("url"),
                        link.get("created") and self._date(link.get("created")) or "",
                    ).rstrip()
                )
            return "\n".join(lines) or "Nenhum payment link encontrado."

        return (
            "Ação desconhecida. Use: overview, revenue, customers, invoices, "
            "payment_links."
        )

    def _register(self) -> None:
        self.registry.register(
            "stripe",
            "Consulta o Stripe: saldo e últimas cobranças, receita do período, clientes, "
            "faturas e payment links. Requer STRIPE_API_KEY.",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "overview",
                            "revenue",
                            "customers",
                            "invoices",
                            "payment_links",
                        ],
                    },
                    "days": {"type": "integer", "description": "Dias para receita (padrão 30)."},
                    "limit": {"type": "integer", "description": "Máximo de itens."},
                    "status": {
                        "type": "string",
                        "description": "Filtro de status de faturas (ex.: paid, open).",
                    },
                    "email": {"type": "string", "description": "Filtrar cliente por e-mail."},
                },
                "required": ["action"],
            },
            self._stripe_tool,
        )
