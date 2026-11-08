"""Modelos de frase, entidades e respostas do dataset sintético de tool calling.

Cada **intenção** tem várias formas de pedir (``phrasings``) e uma função que
sorteia as entidades e devolve ``(slots, resultado)``:

- intenção com ferramenta: ``resultado`` = argumentos da chamada;
- intenção sem ferramenta: ``resultado`` = resposta em texto do assistente.

O id de um modelo de frase é ``<intenção>.<nn>``; a divisão treino/val é feita
por esse id (frases do val nunca aparecem no treino).

Regras de escrita (para não vazar o eval ``evals/data/tool_selection.jsonl``):
entidades próprias (arquivos, URLs, e-mails, repositórios) diferentes das do
eval, e frases escritas do zero. A checagem automática em ``leakage.py`` remove
o que ainda ficar parecido demais; ver ``finetune/README.md``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

Slots = Dict[str, Any]
Make = Callable[[random.Random, "Context"], Tuple[Slots, Any]]


@dataclass
class Context:
    """O que o exemplo sabe do "agora" (a mesma data do system prompt)."""

    now: datetime


@dataclass
class Intent:
    id: str
    tool: Optional[str]  # None = responder sem ferramenta
    phrasings: List[str]
    make: Make
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ entidades

TEXT_FILES = [
    "anotacoes/reuniao_quinta.txt", "rascunho_post.md", "notas/livros_2026.md",
    "docs/arquitetura.md", "config/settings.toml", "scripts/deploy.sh", "app/main.py",
    "app/models.py", "CHANGELOG.md", "requirements-dev.txt", "receitas/bolo_cenoura.txt",
    "planejamento/metas_2027.md", "Makefile", "tests/test_api.py", "notas/estudos_sql.md",
    "web/index.html", "api/rotas.py", "docker-compose.yml", "pyproject.toml",
    "diario/semana_42.md", "contatos/fornecedores.txt", "bot/handlers.py",
]
NEW_FILES = [
    "afazeres.md", "ideias_app.txt", "compras_mercado.txt", "notas/podcast.md",
    "metas_semana.md", "rascunhos/email_cliente.txt", "viagem/roteiro_lisboa.md",
    "estudos/flashcards.txt", "presentes_natal.txt", "lembretes_casa.md",
    "receitas/panqueca.txt", "projetos/backlog.md",
]
SHORT_CONTENTS = [
    "pagar a conta de luz até sexta", "1. revisar PR\n2. responder e-mails",
    "leite, pão integral, bananas", "ligar para o encanador amanhã",
    "ideia: app de controle de plantas", "estudar SQL 30 minutos por dia",
    "levar o carro na revisão", "comprar presente da Bia", "reunião com o time às 14h",
    "trocar a senha do roteador", "assistir à palestra de MLOps", "renovar o passaporte",
]
DIRS = [
    "docs", "scripts", "Imagens", "projetos/site", "Desktop", "Músicas", "backups",
    "app", "tests", "fotos/2026", "notas", "config", "Vídeos", "projetos/laudo",
    "Documentos/impostos", "Downloads/instaladores",
]
REPOS = [
    "juliamendes/site-pessoal", "acme/api-pagamentos", "fastapi/fastapi", "ollama/ollama",
    "juliamendes/laudo-llm", "pallets/flask", "huggingface/transformers", "acme/app-mobile",
]
PDFS = [
    "boleto_condominio.pdf", "manual_geladeira.pdf", "artigo_lora.pdf", "edital_concurso.pdf",
    "extrato_outubro.pdf", "prova_estatistica.pdf", "nota_fiscal_notebook.pdf",
    "Documentos/certidao.pdf", "laudo_exame.pdf", "regulamento_estagio.pdf",
]
DOCXS = [
    "curriculo_2026.docx", "plano_de_aula.docx", "relatorio_estagio.docx",
    "carta_recomendacao.docx", "contrato_freela.docx", "tcc_capitulo2.docx",
    "pauta_assembleia.docx", "briefing_cliente.docx",
]
IMAGES = [
    "print_erro.png", "recibo_farmacia.jpg", "lousa_fisica.jpg", "cardapio.png",
    "uploads/placa_rua.jpg", "captura_tela.png", "cupom_mercado.jpg", "slide_aula.png",
]
SHEETS = [
    "gastos_casa.xlsx", "notas_turma.csv", "estoque_loja.xlsx", "horas_freela.csv",
    "metas_vendas.xlsx", "inventario.tsv", "orcamento_viagem.xlsx", "leads_marketing.csv",
]
SEND_FILES = [
    "relatorios/fechamento_trimestre.pdf", "fotos/aniversario.zip", "planilha_horas.xlsx",
    "export/clientes.csv", "apresentacao_final.pptx", "logs/bot.log.gz",
    "musicas/playlist.m3u", "certificado_curso.pdf",
]
URLS = [
    "https://pt.wikipedia.org/wiki/Aprendizado_de_máquina", "https://www.ollama.com/blog",
    "https://g1.globo.com/tecnologia/", "https://huggingface.co/docs/peft/index",
    "https://blog.exemplo.dev/lora-na-pratica", "https://www.python.org/downloads/",
    "https://receitas.exemplo.com/lasanha", "https://arxiv.org/abs/2106.09685",
]
TOPICS = [
    "aprendizado por reforço", "receita de pão de fermentação natural", "Kubernetes",
    "viagem para o Chile", "orçamento doméstico", "MLOps", "treino de corrida",
    "investimentos em renda fixa", "a reforma da cozinha", "o TCC", "inglês para entrevistas",
    "LoRA e QLoRA", "o projeto Atlas", "a mudança de apartamento", "jardinagem",
]
PEOPLE = ["Carla", "Rafael", "Bia", "Marcos", "a Dra. Helena", "o Thiago", "Lucas", "Fernanda"]
EMAILS = [
    "carla.souza@exemplo.com", "rafael@startup.io", "financeiro@acme.com.br",
    "bia.lima@gmail.com", "suporte@loja.exemplo", "marcos.r@empresa.org",
]
TASKS = [
    "beber água", "tirar a roupa do varal", "ligar para a minha mãe", "pagar o boleto",
    "alongar", "responder o e-mail do Rafael", "buscar a encomenda", "regar as plantas",
    "tomar o antibiótico", "entrar na reunião", "comprar pão", "desligar o forno",
    "mandar o relatório", "renovar o seguro do carro",
]
CITIES = ["Recife", "Curitiba", "São Luís", "Lisboa", "Porto Alegre", "Manaus", "Belo Horizonte"]
SEARCH_QUERIES = [
    "melhores práticas de LoRA para modelos pequenos", "resultado do jogo do Flamengo",
    "preço do Raspberry Pi 5 no Brasil", "lançamentos de filmes desta semana",
    "como funciona o Pix agendado", "notícias sobre a taxa Selic",
    "documentação do Unsloth", "horário de funcionamento do Detran",
]
FUNCS = ["load_config", "send_message", "calcular_frete", "parse_args", "build_index", "get_user"]
PATTERNS = ["FIXME", "HACK", "api_key", "print(", "deprecated", "localhost:8000"]
BRANCHES = ["feature/login-google", "fix/timeout-ollama", "docs/readme-lora", "exp/finetune"]
COMMIT_MSGS = [
    "adiciona testes do parser", "corrige timeout do cliente", "atualiza dependências",
    "melhora mensagens de erro", "remove código morto",
]
WEEKDAYS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def pick(rng: random.Random, pool: List[Any]) -> Any:
    return pool[rng.randrange(len(pool))]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _tomorrow_at(ctx: Context, hour: int, minute: int = 0) -> datetime:
    day = ctx.now + timedelta(days=1)
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ------------------------------------------------------ intenções com ferramenta

SHELL_COMMANDS = [
    "npm test", "npm run build", "make build", "docker ps", "ls -la", "uname -a",
    "cargo build", "go test ./...", "python -m unittest", "whoami", "pip list",
    "docker compose up -d", "node scripts/seed.js", "pytest -x tests/test_api.py",
]
SHELL_NATURAL = [
    ("quanto de memória RAM está livre agora?", "free -h"),
    ("há quanto tempo o computador está ligado?", "uptime"),
    ("qual a versão do Node instalada?", "node --version"),
    ("quais processos do python estão rodando?", "ps aux | grep python"),
    ("como está o uso da GPU?", "nvidia-smi"),
    ("quanto espaço ocupa cada pasta aqui dentro?", "du -sh *"),
    ("quais containers docker estão de pé?", "docker ps"),
    ("qual o IP desta máquina na rede?", "hostname -I"),
    ("mostra as variáveis de ambiente do terminal", "env"),
    ("qual versão do pip eu tenho?", "pip --version"),
]
PACKAGES = [("pip", "pandas"), ("pip", "httpx"), ("npm", "axios"), ("pip", "rich"),
            ("npm", "dayjs"), ("pip", "polars"), ("npm", "zod"), ("pip", "typer")]


def _shell_cmd(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    cmd = pick(rng, SHELL_COMMANDS)
    return {"cmd": cmd}, {"command": cmd}


def _shell_cwd(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    cmd, folder = pick(rng, SHELL_COMMANDS), pick(rng, ["projetos/site", "app", "projetos/laudo", "scripts"])
    return {"cmd": cmd, "dir": folder}, {"command": cmd, "cwd": folder}


def _shell_natural(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    question, cmd = pick(rng, SHELL_NATURAL)
    return {"question": question}, {"command": cmd}


def _shell_install(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    manager, package = pick(rng, PACKAGES)
    return {"mgr": manager, "pkg": package}, {"command": "{} install {}".format(manager, package)}


def _read_file(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, TEXT_FILES)
    return {"file": path}, {"path": path}


def _read_file_lines(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, TEXT_FILES)
    start = rng.choice([1, 5, 20, 40, 100])
    end = start + rng.choice([9, 19, 29, 49])
    return {"file": path, "a": start, "b": end}, {"path": path, "start_line": start, "end_line": end}


def _read_file_head(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path, n = pick(rng, TEXT_FILES), rng.choice([10, 15, 20, 30])
    return {"file": path, "n": n}, {"path": path, "start_line": 1, "end_line": n}


def _write_file(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path, content = pick(rng, NEW_FILES), pick(rng, SHORT_CONTENTS)
    return {"file": path, "content": content}, {"path": path, "content": content}


REPLACEMENTS = [
    ("config/settings.toml", "port = 8000", "port = 8080"),
    ("app/main.py", "DEBUG = True", "DEBUG = False"),
    ("docker-compose.yml", "postgres:14", "postgres:16"),
    ("CHANGELOG.md", "v1.2.0", "v1.3.0"),
    ("web/index.html", "Bem vindo", "Bem-vindo"),
    ("viagem/roteiro_lisboa.md", "sábado", "domingo"),
    ("pyproject.toml", "python = \"^3.8\"", "python = \"^3.10\""),
    ("api/rotas.py", "timeout=10", "timeout=30"),
    ("notas/estudos_sql.md", "JOIN interno", "INNER JOIN"),
    ("bot/handlers.py", "Olá, mundo", "Olá, Julia"),
]


def _replace(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path, old, new = pick(rng, REPLACEMENTS)
    return {"file": path, "old": old, "new": new}, {"path": path, "old": old, "new": new}


def _replace_all(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path, old, new = pick(rng, REPLACEMENTS)
    return {"file": path, "old": old, "new": new}, {"path": path, "old": old, "new": new, "all": True}


def _list_dir(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    folder = pick(rng, DIRS)
    return {"dir": folder}, {"path": folder}


def _list_workspace(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {}


def _search_pattern(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    pattern = pick(rng, PATTERNS)
    return {"pat": pattern}, {"pattern": pattern}


def _search_func(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    func = pick(rng, FUNCS)
    return {"fn": func}, {"pattern": func}


def _search_glob(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    pattern = pick(rng, PATTERNS + FUNCS)
    ext = pick(rng, ["py", "js", "md", "ts", "toml"])
    return {"pat": pattern, "ext": ext}, {"pattern": pattern, "glob": "*." + ext}


def _search_in_dir(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    pattern, folder = pick(rng, PATTERNS + FUNCS), pick(rng, ["app", "scripts", "tests", "bot", "api"])
    return {"pat": pattern, "dir": folder}, {"pattern": pattern, "path": folder}


def _find_ext(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    ext, label = pick(rng, [("mp4", "vídeos .mp4"), ("docx", "documentos do Word"),
                            ("png", "imagens PNG"), ("zip", "arquivos .zip"),
                            ("csv", "CSVs"), ("py", "scripts Python"), ("epub", "livros .epub")])
    folder = pick(rng, DIRS)
    return {"label": label, "dir": folder}, {"name": "*." + ext, "path": folder}


def _find_name(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    name = pick(rng, ["declaracao_ir_2025.pdf", "chave_ssh.pub", "foto_passaporte.jpg",
                      "backup_banco.sql", "senhas_wifi.txt", "notas_fiscais.zip", "logo_vetor.svg"])
    return {"fname": name}, {"name": name}


def _find_prefix(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    prefix = pick(rng, ["test_", "relatorio_", "IMG_", "fatura_", "draft_"])
    return {"pre": prefix}, {"name": prefix + "*"}


def _git_status(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"args": ["status"]}


def _git_log(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    n = rng.choice([3, 7, 10, 15])
    return {"n": n}, {"args": ["log", "--oneline", "-n", str(n)]}


def _git_diff_file(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, TEXT_FILES)
    return {"file": path}, {"args": ["diff", path]}


def _git_branch(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"args": ["branch", "--show-current"]}


def _git_new_branch(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    branch = pick(rng, BRANCHES)
    return {"b": branch}, {"args": ["checkout", "-b", branch]}


def _git_commit(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    msg = pick(rng, COMMIT_MSGS)
    return {"msg": msg}, {"args": ["commit", "-m", msg]}


def _git_add(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, TEXT_FILES)
    return {"file": path}, {"args": ["add", path]}


def _git_pull(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"args": ["pull"]}


def _gh_repo(action: str) -> Make:
    def make(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
        repo = pick(rng, REPOS)
        return {"repo": repo}, {"action": action, "repo": repo}
    return make


def _gh_number(action: str) -> Make:
    def make(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
        repo, number = pick(rng, REPOS), rng.randint(2, 480)
        return {"repo": repo, "n": number}, {"action": action, "repo": repo, "number": number}
    return make


def _gh_create_issue(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    repo = pick(rng, REPOS)
    title = pick(rng, ["Erro ao salvar o perfil", "Adicionar modo escuro", "Timeout no login",
                       "Documentar a API de pagamentos", "Melhorar o tempo de build"])
    return {"repo": repo, "title": title}, {"action": "create_issue", "repo": repo, "title": title}


def _gh_comment(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    repo, number = pick(rng, REPOS), rng.randint(2, 480)
    body = pick(rng, ["consegui reproduzir no Windows", "corrigido no último commit",
                      "posso pegar essa", "precisa de mais detalhes para reproduzir"])
    return ({"repo": repo, "n": number, "body": body},
            {"action": "comment_issue", "repo": repo, "number": number, "body": body})


def _gh_closed(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    repo = pick(rng, REPOS)
    return {"repo": repo}, {"action": "list_issues", "repo": repo, "state": "closed"}


def _gh_list_repos(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"action": "list_repos"}


OBSIDIAN_NOTES = ["Diário/2026-11.md", "Leituras.md", "Projetos/Atlas.md", "Receitas/Bolos.md",
                  "Treinos.md", "Ideias de negócio.md", "Cursos/MLOps.md", "Viagens/Chile.md"]
OBSIDIAN_FOLDERS = ["Diário", "Projetos", "Receitas", "Cursos", "Viagens", "Inbox"]


def _obs_search(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    topic = pick(rng, TOPICS)
    return {"topic": topic}, {"action": "search", "query": topic}


def _obs_read(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    note = pick(rng, OBSIDIAN_NOTES)
    return {"note": note}, {"action": "read", "path": note}


def _obs_append(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    note, content = pick(rng, OBSIDIAN_NOTES), pick(rng, SHORT_CONTENTS)
    return ({"note": note, "content": content},
            {"action": "write", "path": note, "content": content, "mode": "append"})


def _obs_overwrite(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    note, content = pick(rng, OBSIDIAN_NOTES), pick(rng, SHORT_CONTENTS)
    return ({"note": note, "content": content},
            {"action": "write", "path": note, "content": content, "mode": "overwrite"})


def _obs_list(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    folder = pick(rng, OBSIDIAN_FOLDERS)
    return {"folder": folder}, {"action": "list", "folder": folder}


NOTION_IDS = ["1a2b3c4d5e6f", "9f8e7d6c5b4a", "a1b2c3d4e5f6", "0c9d8e7f6a5b"]


def _notion_search(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    topic = pick(rng, TOPICS)
    return {"topic": topic}, {"action": "search", "query": topic}


def _notion_read(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    page = pick(rng, NOTION_IDS)
    return {"id": page}, {"action": "read", "page_id": page}


def _notion_append(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    page, content = pick(rng, NOTION_IDS), pick(rng, SHORT_CONTENTS)
    return {"id": page, "content": content}, {"action": "append", "page_id": page, "content": content}


def _notion_create(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    title = pick(rng, ["Metas de 2027", "Leituras de dezembro", "Checklist da mudança",
                       "Roteiro do podcast", "Retrospectiva do sprint", "Receitas favoritas"])
    return {"title": title}, {"action": "create_page", "title": title}


def _sheet_read(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, SHEETS)
    return {"file": path}, {"action": "read", "path": path}


def _sheet_tab(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, [s for s in SHEETS if s.endswith(".xlsx")])
    sheet = pick(rng, ["Janeiro", "Resumo", "Q3", "Dados brutos", "2025"])
    return {"file": path, "sheet": sheet}, {"action": "read", "path": path, "sheet": sheet}


def _sheet_rows(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path, rows = pick(rng, SHEETS), rng.choice([10, 20, 50])
    return {"file": path, "n": rows}, {"action": "read", "path": path, "max_rows": rows}


EVENTS = ["dentista", "almoço com a Carla", "call com o cliente", "aula de inglês",
          "revisão do carro", "academia", "entrevista de emprego", "aniversário do Lucas"]


def _gcal_list(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    label, days = pick(rng, [("hoje", 1), ("nos próximos 2 dias", 2), ("nesta semana", 7),
                             ("nos próximos 5 dias", 5), ("nas próximas duas semanas", 14)])
    return {"period": label}, {"action": "calendar_list", "days": days}


def _cal_create(action_tool: str) -> Make:
    def make(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
        event, hour = pick(rng, EVENTS), rng.choice([8, 10, 13, 16, 19])
        start = _tomorrow_at(ctx, hour)
        return ({"event": event, "hour": hour},
                {"action": "calendar_create", "summary": event, "start": _iso(start),
                 "end": _iso(start + timedelta(hours=1))})
    return make


def _gmail_from(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    sender = pick(rng, ["Nubank", "Amazon", "LinkedIn", "Detran", "faculdade", "Mercado Livre"])
    return {"sender": sender}, {"action": "gmail_search", "query": "from:" + sender.lower().replace(" ", "")}


def _gmail_unread(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"action": "gmail_search", "query": "is:unread"}


def _gmail_send(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    to = pick(rng, EMAILS)
    subject, body = pick(rng, [("Proposta", "segue a proposta revisada"),
                               ("Reunião", "podemos remarcar para quinta?"),
                               ("Pagamento", "o pagamento foi feito hoje"),
                               ("Documentos", "já enviei os documentos")])
    return ({"to": to, "subj": subject, "body": body},
            {"action": "gmail_send", "to": to, "subject": subject, "body": body})


def _ms_mail_list(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"action": "mail_list"}


def _ms_mail_search(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    query = pick(rng, ["nota fiscal", "férias", "contrato", "orçamento do projeto", "treinamento"])
    return {"q": query}, {"action": "mail_search", "query": query}


def _ms_mail_send(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    to = pick(rng, EMAILS)
    subject, body = pick(rng, [("Status", "o deploy terminou sem erros"),
                               ("Atraso", "vou chegar 15 minutos atrasada"),
                               ("Planilha", "a planilha foi atualizada")])
    return ({"to": to, "subj": subject, "body": body},
            {"action": "mail_send", "to": to, "subject": subject, "body": body})


def _ms_cal_list(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    label, days = pick(rng, [("hoje", 1), ("nos próximos 3 dias", 3), ("nesta semana", 7)])
    return {"period": label}, {"action": "calendar_list", "days": days}


def _ms_simple(action: str) -> Make:
    def make(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
        return {}, {"action": action}
    return make


def _stripe_simple(action: str) -> Make:
    def make(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
        return {}, {"action": action}
    return make


def _stripe_revenue(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    days = rng.choice([7, 15, 60, 90])
    return {"n": days}, {"action": "revenue", "days": days}


def _stripe_open_invoices(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"action": "invoices", "status": "open"}


def _stripe_customer(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    email = pick(rng, EMAILS)
    return {"email": email}, {"action": "customers", "email": email}


def _web_search(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    query = pick(rng, SEARCH_QUERIES)
    return {"q": query}, {"query": query}


def _web_weather(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    city = pick(rng, CITIES)
    return {"city": city}, {"query": "previsão do tempo " + city}


def _web_fetch(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    url = pick(rng, URLS)
    return {"url": url}, {"url": url}


def _send_file(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, SEND_FILES)
    return {"file": path}, {"path": path}


def _read_pdf(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, PDFS)
    return {"file": path}, {"path": path}


def _read_pdf_pages(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path, pages = pick(rng, PDFS), rng.choice([2, 3, 5, 10])
    return {"file": path, "n": pages}, {"path": path, "max_pages": pages}


def _read_docx(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, DOCXS)
    return {"file": path}, {"path": path}


def _read_image(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    path = pick(rng, IMAGES)
    return {"file": path}, {"path": path}


SPEAK_TEXTS = ["a pizza chegou", "não esquece o guarda-chuva", "o deploy terminou",
               "boa sorte na entrevista", "hora de dormir", "o almoço está pronto",
               "a reunião começa em cinco minutos"]


def _speak(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    text = pick(rng, SPEAK_TEXTS)
    return {"text": text}, {"text": text}


def _ks(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    topic = pick(rng, TOPICS)
    return {"topic": topic}, {"query": topic}


def _ks_k(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    topic, k = pick(rng, TOPICS), rng.choice([3, 8, 10])
    return {"topic": topic, "k": k}, {"query": topic, "k": k}


def _ki_path(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    folder = pick(rng, ["Documentos/impostos", "notas", "projetos/laudo", "docs", "Desktop/artigos"])
    return {"dir": folder}, {"path": folder}


def _ki_default(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {}


def _ki_force(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"force": True}


def _rem_minutes(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    n, task = rng.choice([5, 15, 20, 25, 30, 45]), pick(rng, TASKS)
    return {"n": n, "task": task}, {"action": "add", "when": "{}m".format(n), "text": task}


def _rem_hours(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    n, task = rng.choice([1, 2, 3, 4]), pick(rng, TASKS)
    return {"n": n, "task": task}, {"action": "add", "when": "{}h".format(n), "text": task}


def _rem_tomorrow(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    hour, task = rng.choice([7, 8, 10, 14, 18, 21]), pick(rng, TASKS)
    return ({"hh": hour, "task": task},
            {"action": "add", "when": "amanhã {:02d}:00".format(hour), "text": task})


def _rem_today(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    hour, minute, task = rng.choice([17, 18, 20, 22]), rng.choice([0, 15, 30, 45]), pick(rng, TASKS)
    return ({"hh": hour, "mm": "{:02d}".format(minute), "task": task},
            {"action": "add", "when": "hoje {:02d}:{:02d}".format(hour, minute), "text": task})


def _rem_list(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    return {}, {"action": "list"}


def _rem_delete(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    rid = rng.randint(1, 40)
    return {"id": rid}, {"action": "delete", "id": rid}


TOOL_INTENTS: List[Intent] = [
    # run_shell
    Intent("run_shell.cmd", "run_shell", [
        "executa {cmd} no terminal", "roda `{cmd}` pra mim", "pode rodar o comando {cmd}?",
        "manda um {cmd} aí", "no terminal: {cmd}", "dispara o {cmd} e me fala o resultado",
        "quero ver a saída de {cmd}", "roda aí: {cmd}",
    ], _shell_cmd),
    Intent("run_shell.cwd", "run_shell", [
        "entra em {dir} e roda {cmd}", "roda {cmd} dentro da pasta {dir}",
        "na pasta {dir}, executa {cmd}", "executa `{cmd}` com o diretório {dir}",
    ], _shell_cwd, 0.6),
    Intent("run_shell.natural", "run_shell", [
        "{question}", "dá uma olhada no terminal: {question}", "checa pra mim: {question}",
        "consulta na máquina: {question}",
    ], _shell_natural),
    Intent("run_shell.install", "run_shell", [
        "adiciona a dependência {pkg} via {mgr}", "faz um {mgr} install do {pkg}",
        "preciso do pacote {pkg}, instala com {mgr}", "coloca o {pkg} no projeto usando {mgr}",
        "baixa e instala o {pkg} pelo {mgr}",
    ], _shell_install, 0.7),
    # read_file
    Intent("read_file.whole", "read_file", [
        "mostra o conteúdo de {file}", "lê o {file} pra mim", "o que está escrito em {file}?",
        "me mostra o arquivo {file}", "dá uma olhada no {file}", "exibe o {file}",
        "quero ler o arquivo {file}", "imprime o conteúdo do {file}",
    ], _read_file),
    Intent("read_file.lines", "read_file", [
        "lê do trecho {a} até a linha {b} de {file}", "mostra as linhas {a}-{b} do {file}",
        "quero ver de {a} a {b} no arquivo {file}", "abre {file} entre as linhas {a} e {b}",
    ], _read_file_lines, 0.6),
    Intent("read_file.head", "read_file", [
        "mostra as primeiras {n} linhas de {file}", "lê só o começo do {file}, umas {n} linhas",
        "me mostra o topo do {file} (primeiras {n} linhas)",
    ], _read_file_head, 0.4),
    # write_file
    Intent("write_file.new", "write_file", [
        "cria o arquivo {file} com o conteúdo: {content}", "salva '{content}' em {file}",
        "escreve num arquivo {file}: {content}", "gera um arquivo {file} contendo \"{content}\"",
        "anota em {file} o seguinte: {content}", "guarda isso no arquivo {file}: {content}",
        "faz um {file} novo com o texto {content}",
    ], _write_file),
    # replace_in_file
    Intent("replace_in_file.one", "replace_in_file", [
        "em {file}, troca '{old}' por '{new}'", "substitui {old} por {new} no arquivo {file}",
        "no {file} muda '{old}' para '{new}'", "edita o {file}: onde está '{old}' coloca '{new}'",
        "altera no {file} o trecho \"{old}\" para \"{new}\"", "atualiza {file} trocando {old} por {new}",
    ], _replace),
    Intent("replace_in_file.all", "replace_in_file", [
        "troca todas as ocorrências de '{old}' por '{new}' em {file}",
        "em {file}, substitui tudo que for '{old}' por '{new}'",
        "no {file}, onde aparecer {old}, coloca {new} (todas as vezes)",
    ], _replace_all, 0.4),
    # list_dir
    Intent("list_dir.path", "list_dir", [
        "o que tem na pasta {dir}?", "lista o conteúdo de {dir}", "mostra os arquivos dentro de {dir}",
        "quais arquivos estão em {dir}?", "abre a pasta {dir} e me diz o que tem lá",
        "me dá a listagem do diretório {dir}", "dá um ls na pasta {dir}",
    ], _list_dir),
    Intent("list_dir.workspace", "list_dir", [
        "o que tem na pasta de trabalho?", "lista os arquivos do workspace",
        "mostra o que existe no diretório atual",
    ], _list_workspace, 0.3),
    # search_files
    Intent("search_files.pattern", "search_files", [
        "procura '{pat}' nos arquivos", "busca a string {pat} no código",
        "em que arquivos aparece {pat}?", "faz um grep por {pat}", "tem {pat} em algum lugar do código?",
    ], _search_pattern),
    Intent("search_files.func", "search_files", [
        "onde eu chamo a função {fn}?", "quem usa {fn} no projeto?",
        "acha as referências a {fn}", "em que linha está definida a {fn}?",
    ], _search_func),
    Intent("search_files.glob", "search_files", [
        "procura {pat} só nos arquivos .{ext}", "busca {pat} apenas em arquivos *.{ext}",
        "nos .{ext}, onde aparece {pat}?",
    ], _search_glob, 0.5),
    Intent("search_files.dir", "search_files", [
        "procura {pat} dentro da pasta {dir}", "grep por {pat} em {dir}",
        "na pasta {dir}, onde aparece {pat}?",
    ], _search_in_dir, 0.5),
    # find_files
    Intent("find_files.ext", "find_files", [
        "acha os {label} em {dir}", "procura todos os {label} na pasta {dir}",
        "lista os {label} que estão em {dir}", "quais {label} eu tenho em {dir}?",
    ], _find_ext),
    Intent("find_files.name", "find_files", [
        "cadê o arquivo {fname}?", "procura um arquivo com o nome {fname}",
        "em que pasta ficou o {fname}?", "localiza o {fname} no computador",
    ], _find_name),
    Intent("find_files.prefix", "find_files", [
        "procura arquivos que comecem com {pre}", "acha tudo cujo nome começa com {pre}",
        "lista os arquivos com prefixo {pre}",
    ], _find_prefix, 0.5),
    # git
    Intent("git.status", "git", [
        "como está o git?", "tem alguma mudança não commitada?", "git status por favor",
        "o que mudou no repositório desde o último commit?", "tem arquivo modificado no repo?",
    ], _git_status),
    Intent("git.log", "git", [
        "mostra os {n} commits mais recentes", "quais foram os últimos {n} commits?",
        "histórico do git, só os últimos {n}", "lista {n} commits do log",
    ], _git_log),
    Intent("git.diff", "git", [
        "o que mudou no {file} que ainda não foi commitado?", "mostra o diff do {file}",
        "me mostra as alterações em {file}",
    ], _git_diff_file, 0.6),
    Intent("git.branch", "git", [
        "em que branch eu estou?", "qual é a branch atual?", "me diz o nome da branch ativa",
    ], _git_branch, 0.5),
    Intent("git.new_branch", "git", [
        "cria uma branch chamada {b}", "abre uma branch nova {b} e muda pra ela",
        "faz checkout numa branch nova: {b}",
    ], _git_new_branch, 0.5),
    Intent("git.commit", "git", [
        "commita com a mensagem '{msg}'", "faz o commit: {msg}", "registra um commit dizendo \"{msg}\"",
    ], _git_commit, 0.6),
    Intent("git.add", "git", [
        "adiciona o {file} no stage", "faz git add do {file}", "coloca o {file} na área de preparação",
    ], _git_add, 0.5),
    Intent("git.pull", "git", [
        "atualiza o repositório local com o remoto", "puxa as novidades do remoto", "dá um git pull",
    ], _git_pull, 0.4),
    # github
    Intent("github.issues", "github", [
        "quais issues estão abertas em {repo}?", "lista as issues do {repo} no GitHub",
        "me mostra as issues pendentes do repositório {repo}", "tem issue aberta no {repo}?",
    ], _gh_repo("list_issues")),
    Intent("github.prs", "github", [
        "quais PRs estão abertos no {repo}?", "lista os pull requests do {repo}",
        "tem pull request esperando revisão no {repo}?",
    ], _gh_repo("list_prs")),
    Intent("github.get_issue", "github", [
        "mostra a issue #{n} do {repo}", "abre a issue {n} do repositório {repo} no GitHub",
        "o que diz a issue número {n} em {repo}?",
    ], _gh_number("get_issue"), 0.6),
    Intent("github.get_pr", "github", [
        "detalhes do PR #{n} em {repo}", "mostra o pull request {n} do {repo}",
    ], _gh_number("get_pr"), 0.5),
    Intent("github.commits", "github", [
        "últimos commits do {repo} no GitHub", "o que foi commitado recentemente em {repo} no GitHub?",
    ], _gh_repo("recent_commits"), 0.5),
    Intent("github.create_issue", "github", [
        "cria uma issue em {repo} com o título '{title}'", "registra no GitHub ({repo}) uma issue: {title}",
        "abre um chamado no repositório {repo} chamado \"{title}\"",
    ], _gh_create_issue),
    Intent("github.comment", "github", [
        "comenta na issue {n} do {repo}: {body}", "responde na issue #{n} de {repo} dizendo \"{body}\"",
    ], _gh_comment, 0.5),
    Intent("github.closed", "github", [
        "lista as issues fechadas do {repo}", "quais issues já foram resolvidas em {repo}?",
    ], _gh_closed, 0.4),
    Intent("github.repos", "github", [
        "lista meus repositórios no GitHub", "quais repositórios eu tenho no GitHub?",
    ], _gh_list_repos, 0.3),
    Intent("github.repo_info", "github", [
        "me dá as informações do repositório {repo} no GitHub", "quantas estrelas tem o {repo}?",
    ], _gh_repo("repo_info"), 0.4),
    # obsidian
    Intent("obsidian.search", "obsidian", [
        "procura no Obsidian notas sobre {topic}", "no meu vault do Obsidian, o que tem sobre {topic}?",
        "busca {topic} no Obsidian", "acha no Obsidian alguma nota falando sobre {topic}",
    ], _obs_search),
    Intent("obsidian.read", "obsidian", [
        "abre a nota {note} do Obsidian", "lê no Obsidian a nota {note}", "mostra a nota {note} do vault",
    ], _obs_read),
    Intent("obsidian.append", "obsidian", [
        "adiciona na nota {note} do Obsidian: {content}", "no Obsidian, acrescenta em {note} o texto \"{content}\"",
        "anota no fim da nota {note} (Obsidian): {content}",
    ], _obs_append),
    Intent("obsidian.overwrite", "obsidian", [
        "reescreve a nota {note} do Obsidian só com: {content}", "substitui todo o conteúdo de {note} no Obsidian por \"{content}\"",
    ], _obs_overwrite, 0.4),
    Intent("obsidian.list", "obsidian", [
        "quais notas tenho na pasta {folder} do Obsidian?", "lista o que tem em {folder} no Obsidian",
        "mostra as notas da pasta {folder} do meu vault",
    ], _obs_list),
    # notion
    Intent("notion.search", "notion", [
        "procura no Notion a página sobre {topic}", "busca {topic} no Notion",
        "tem alguma página no Notion falando sobre {topic}?",
    ], _notion_search),
    Intent("notion.read", "notion", [
        "lê a página {id} do Notion", "abre no Notion a página de id {id}",
    ], _notion_read, 0.6),
    Intent("notion.append", "notion", [
        "acrescenta na página {id} do Notion: {content}", "adiciona o texto \"{content}\" na página {id} do Notion",
    ], _notion_append, 0.6),
    Intent("notion.create", "notion", [
        "cria uma página no Notion com o título '{title}'", "faz uma página nova no Notion: {title}",
        "abre no Notion uma página \"{title}\"",
    ], _notion_create),
    # spreadsheet
    Intent("spreadsheet.read", "spreadsheet", [
        "lê a planilha {file}", "abre o {file} e me resume", "o que tem na planilha {file}?",
        "mostra os dados de {file}", "analisa a planilha {file}",
    ], _sheet_read),
    Intent("spreadsheet.tab", "spreadsheet", [
        "abre a aba {sheet} do {file}", "na planilha {file}, lê a aba {sheet}",
        "mostra a guia {sheet} de {file}",
    ], _sheet_tab, 0.7),
    Intent("spreadsheet.rows", "spreadsheet", [
        "mostra as primeiras {n} linhas da planilha {file}", "lê só {n} linhas de {file}",
    ], _sheet_rows, 0.5),
    # google
    Intent("google.cal_list", "google", [
        "o que tem na minha agenda do Google {period}?", "meus compromissos do Google Agenda {period}",
        "tenho algo marcado no Google Calendar {period}?", "lista os eventos da agenda do Google {period}",
    ], _gcal_list),
    Intent("google.cal_create", "google", [
        "agenda no Google Calendar {event} amanhã às {hour}h", "cria no Google Agenda: {event}, amanhã {hour}h",
        "coloca na minha agenda do Google {event} para amanhã às {hour} horas",
    ], _cal_create("google")),
    Intent("google.gmail_from", "google", [
        "tem e-mail do {sender} no Gmail?", "procura no Gmail mensagens do {sender}",
        "busca os e-mails que o {sender} mandou no Gmail",
    ], _gmail_from),
    Intent("google.gmail_unread", "google", [
        "quais e-mails não lidos tenho no Gmail?", "chegou algo novo no Gmail que eu não li?",
    ], _gmail_unread, 0.5),
    Intent("google.gmail_send", "google", [
        "manda um e-mail pelo Gmail para {to} com o assunto '{subj}' dizendo: {body}",
        "envia do meu Gmail para {to}, assunto {subj}: {body}",
    ], _gmail_send, 0.7),
    # microsoft
    Intent("microsoft.mail_list", "microsoft", [
        "chegou e-mail novo no Outlook?", "lista os e-mails da minha caixa do Outlook",
        "tem mensagem nova no e-mail do trabalho (Outlook)?",
    ], _ms_mail_list),
    Intent("microsoft.mail_search", "microsoft", [
        "procura no Outlook e-mails sobre {q}", "busca \"{q}\" nos e-mails do Outlook",
        "acha no Outlook a mensagem que fala de {q}",
    ], _ms_mail_search),
    Intent("microsoft.mail_send", "microsoft", [
        "envia pelo Outlook um e-mail para {to}, assunto '{subj}': {body}",
        "pelo e-mail do trabalho, manda para {to} (assunto {subj}) dizendo {body}",
    ], _ms_mail_send, 0.7),
    Intent("microsoft.cal_list", "microsoft", [
        "minha agenda do Outlook {period}", "que reuniões tenho no calendário do Outlook {period}?",
        "lista meus eventos do Microsoft 365 {period}",
    ], _ms_cal_list),
    Intent("microsoft.cal_create", "microsoft", [
        "marca no calendário do Outlook {event} amanhã às {hour}h",
        "cria um evento no Outlook: {event}, amanhã {hour}h",
    ], _cal_create("microsoft"), 0.6),
    Intent("microsoft.teams_chats", "microsoft", [
        "mostra meus chats do Teams", "quais conversas tenho no Microsoft Teams?",
    ], _ms_simple("teams_chats"), 0.5),
    Intent("microsoft.teams_list", "microsoft", [
        "de quais equipes eu participo no Teams?", "lista minhas equipes do Teams",
    ], _ms_simple("teams_list"), 0.4),
    Intent("microsoft.me", "microsoft", [
        "qual é o meu perfil no Microsoft 365?", "com qual conta estou logada no Microsoft 365?",
    ], _ms_simple("me"), 0.3),
    # stripe
    Intent("stripe.overview", "stripe", [
        "como está meu saldo no Stripe?", "me dá um resumo da conta do Stripe",
        "quanto tenho disponível no Stripe?",
    ], _stripe_simple("overview")),
    Intent("stripe.revenue", "stripe", [
        "quanto entrou no Stripe nos últimos {n} dias?", "receita do Stripe dos últimos {n} dias",
        "faturamento no Stripe em {n} dias",
    ], _stripe_revenue),
    Intent("stripe.customers", "stripe", [
        "lista os clientes do Stripe", "quem são meus clientes no Stripe?",
    ], _stripe_simple("customers"), 0.5),
    Intent("stripe.customer_email", "stripe", [
        "procura no Stripe o cliente {email}", "acha o cadastro de {email} no Stripe",
    ], _stripe_customer, 0.5),
    Intent("stripe.open_invoices", "stripe", [
        "quais faturas estão em aberto no Stripe?", "tem fatura pendente no Stripe?",
    ], _stripe_open_invoices, 0.6),
    Intent("stripe.links", "stripe", [
        "mostra meus links de pagamento do Stripe", "quais payment links eu tenho no Stripe?",
    ], _stripe_simple("payment_links"), 0.4),
    # web
    Intent("web_search.query", "web_search", [
        "pesquisa na internet: {q}", "busca na web {q}", "procura online sobre {q}",
        "dá uma pesquisada: {q}", "o que a internet diz sobre {q}?",
    ], _web_search),
    Intent("web_search.weather", "web_search", [
        "vai chover em {city} amanhã?", "como está o tempo em {city}?", "previsão do tempo para {city}",
    ], _web_weather, 0.6),
    Intent("web_fetch.url", "web_fetch", [
        "lê essa página: {url}", "o que diz {url}?", "abre {url} e me conta o conteúdo",
        "faz um resumo do link {url}", "pega o texto de {url}",
    ], _web_fetch),
    # telegram / documentos / voz
    Intent("send_file.path", "send_file", [
        "me manda o arquivo {file}", "envia {file} aqui no chat", "manda pra mim no Telegram o {file}",
        "quero receber o {file} por aqui", "anexa o {file} na conversa",
    ], _send_file),
    Intent("read_pdf.path", "read_pdf", [
        "lê o PDF {file}", "o que diz o {file}?", "extrai o texto do {file}", "resume o PDF {file}",
        "abre o {file} e me diz o principal",
    ], _read_pdf),
    Intent("read_pdf.pages", "read_pdf", [
        "lê só as {n} primeiras páginas de {file}", "do {file}, quero o texto das primeiras {n} páginas",
    ], _read_pdf_pages, 0.5),
    Intent("read_docx.path", "read_docx", [
        "abre o documento {file}", "lê o Word {file}", "o que está escrito no {file}?",
        "extrai o texto do arquivo {file}", "me resume o {file}",
    ], _read_docx),
    Intent("read_image.path", "read_image", [
        "faz OCR da imagem {file}", "lê o texto do print {file}", "o que está escrito na foto {file}?",
        "transcreve o texto da imagem {file}", "extrai as letras de {file}",
    ], _read_image),
    Intent("speak.text", "speak", [
        "manda um áudio dizendo: {text}", "fala em voz alta: {text}", "me responde em áudio: {text}",
        "grava um áudio com '{text}'", "quero ouvir isso em voz: {text}",
    ], _speak),
    # memória local
    Intent("knowledge_search.topic", "knowledge_search", [
        "procura nos meus documentos o que tem sobre {topic}", "nas minhas notas, o que eu escrevi sobre {topic}?",
        "busca na minha base de conhecimento: {topic}", "o que os meus arquivos dizem sobre {topic}?",
        "eu tinha guardado algo sobre {topic}, acha pra mim", "consulta a memória local sobre {topic}",
    ], _ks),
    Intent("knowledge_search.k", "knowledge_search", [
        "traz os {k} trechos mais relevantes sobre {topic} da minha base",
        "busca {topic} nos meus documentos e me dá {k} resultados",
    ], _ks_k, 0.5),
    Intent("knowledge_index.path", "knowledge_index", [
        "indexa a pasta {dir} na memória", "adiciona {dir} à busca semântica",
        "coloca os arquivos de {dir} no índice", "inclui a pasta {dir} na base de conhecimento",
    ], _ki_path),
    Intent("knowledge_index.default", "knowledge_index", [
        "atualiza o índice da memória local", "roda a indexação das pastas padrão",
        "reprocessa a base de conhecimento",
    ], _ki_default, 0.6),
    Intent("knowledge_index.force", "knowledge_index", [
        "reindexa tudo do zero, mesmo o que não mudou", "força a reindexação completa da memória",
    ], _ki_force, 0.4),
    # lembretes
    Intent("reminder.minutes", "reminder", [
        "me lembra de {task} em {n} minutos", "daqui a {n} min me avisa pra {task}",
        "em {n} minutos me lembra: {task}", "cria um lembrete de {task} para daqui {n} minutos",
    ], _rem_minutes),
    Intent("reminder.hours", "reminder", [
        "me avisa em {n}h para {task}", "daqui a {n} horas me lembra de {task}",
        "lembrete em {n} horas: {task}",
    ], _rem_hours, 0.7),
    Intent("reminder.tomorrow", "reminder", [
        "amanhã às {hh}h me lembra de {task}", "cria um lembrete para amanhã {hh}h: {task}",
        "me lembra amanhã, {hh} horas, de {task}",
    ], _rem_tomorrow),
    Intent("reminder.today", "reminder", [
        "hoje às {hh}:{mm} me lembra de {task}", "lembra-me de {task} hoje {hh}:{mm}",
    ], _rem_today, 0.6),
    Intent("reminder.list", "reminder", [
        "quais lembretes estão marcados?", "lista meus lembretes", "tenho algum lembrete agendado?",
    ], _rem_list, 0.6),
    Intent("reminder.delete", "reminder", [
        "apaga o lembrete {id}", "remove o lembrete #{id}", "não preciso mais do lembrete {id}, cancela",
    ], _rem_delete, 0.6),
]


# ---------------------------------------------------- intenções sem ferramenta


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        text = "{:,}".format(int(round(value))).replace(",", ".")
    else:
        text = "{:,.2f}".format(value).replace(",", "X").replace(".", ",").replace("X", ".")
    return text


def _reply_pool(pool: List[str]) -> Make:
    def make(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
        return {}, pick(rng, pool)
    return make


GREETING_REPLIES = [
    "Oi, Julia! Tudo certo por aqui. Em que posso ajudar?",
    "Olá! Pronto para ajudar. O que você precisa?",
    "Oi! Tudo tranquilo. Quer que eu faça algo?",
    "Bom te ver por aqui! Diz aí no que posso ajudar.",
]
THANKS_REPLIES = [
    "De nada! Se precisar de mais alguma coisa, é só chamar.",
    "Imagina! Estou por aqui.",
    "Por nada, Julia. Até a próxima!",
    "Disponha! Qualquer coisa, me chama.",
]


def _percent(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    p, n = rng.choice([5, 8, 12, 20, 25, 30, 40]), rng.choice([80, 150, 360, 500, 720, 1200, 2500])
    return {"p": p, "n": n}, "{}% de {} é {}.".format(p, _fmt_num(n), _fmt_num(p * n / 100))


def _multiply(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    a, b = rng.randint(11, 99), rng.randint(3, 49)
    return {"a": a, "b": b}, "{} × {} = {}.".format(a, b, _fmt_num(a * b))


def _money(value: float) -> str:
    """1234.5 -> '1.234,50' (formato brasileiro, duas casas)."""
    return "{:,.2f}".format(value).replace(",", "X").replace(".", ",").replace("X", ".")


def _split_bill(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    total, people = rng.choice([90, 147, 236, 312, 480]), rng.choice([2, 3, 4, 6])
    return ({"v": total, "k": people},
            "R$ {} divididos por {} dá R$ {} para cada um.".format(
                _money(total), people, _money(total / people)))


def _tip(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    total = rng.choice([84, 120, 156, 230, 98])
    tip = total * 0.10
    return ({"v": total},
            "Com 10% de serviço: R$ {} de gorjeta, total de R$ {}.".format(_money(tip), _money(total + tip)))


def _celsius(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    c = rng.choice([-5, 0, 18, 25, 32, 37, 40])
    f = c * 9 / 5 + 32
    f_text = "{:.1f}".format(f).replace(".", ",").replace(",0", "")
    return {"c": c}, "{} °C = {} °F (F = C × 9/5 + 32).".format(c, f_text)


def _km_miles(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    km = rng.choice([5, 10, 21, 42, 100])
    miles = km / 1.609344
    return {"km": km}, "{} km são cerca de {} milhas.".format(km, "{:.1f}".format(miles).replace(".", ","))


EXPLANATIONS = [
    ("o que é um commit no git?", "Um commit é um registro de um conjunto de mudanças no repositório, com autor, data e mensagem. Ele vira um ponto do histórico ao qual você pode voltar."),
    ("pra que serve o comando ls no Linux?", "O `ls` lista arquivos e pastas de um diretório. Com `-l` mostra detalhes (permissões, tamanho, data) e com `-a` inclui os ocultos."),
    ("o que significa erro HTTP 404?", "404 quer dizer \"não encontrado\": o servidor respondeu, mas o endereço pedido não existe."),
    ("qual a diferença entre memória RAM e armazenamento?", "A RAM é a memória rápida e temporária usada pelos programas abertos; some ao desligar. O armazenamento (SSD/HD) guarda arquivos de forma permanente."),
    ("o que é machine learning, em uma frase?", "É fazer o computador aprender padrões a partir de exemplos, em vez de seguir regras escritas à mão."),
    ("o que é LoRA em modelos de linguagem?", "LoRA (Low-Rank Adaptation) ajusta um modelo grande treinando só pequenas matrizes extras de baixo posto, em vez de todos os pesos. Fica barato e o adaptador ocupa poucos MB."),
    ("como funciona o pip do Python?", "O `pip` baixa pacotes do PyPI e instala no ambiente Python ativo, resolvendo as dependências. Ex.: `pip install nome-do-pacote`."),
    ("o que é um arquivo .env?", "É um arquivo de texto com variáveis de ambiente (chave=valor), usado para guardar configurações e segredos fora do código. Não deve ir para o git."),
    ("explica Docker em poucas palavras", "Docker empacota uma aplicação com tudo que ela precisa (bibliotecas, configuração) num contêiner, que roda igual em qualquer máquina."),
    ("o que é recursão em programação?", "É quando uma função chama a si mesma para resolver partes menores do problema, até chegar num caso base que para a repetição."),
    ("o que é JSON?", "JSON é um formato de texto para dados estruturados, com objetos `{\"chave\": valor}` e listas `[...]`. É o padrão de troca de dados em APIs."),
    ("o que faz o git stash?", "O `git stash` guarda temporariamente as mudanças não commitadas e limpa a área de trabalho. Depois, `git stash pop` traz tudo de volta."),
    ("qual a diferença entre CC e CCO num e-mail?", "Em CC os destinatários veem quem mais recebeu a cópia; em CCO (cópia oculta) ninguém vê quem está nessa lista."),
    ("o que é o horário UTC?", "UTC é o tempo universal coordenado, a referência mundial de horário. Brasília fica em UTC−3."),
    ("como um PDF escaneado é diferente de um PDF normal?", "O escaneado é uma imagem da página, sem texto selecionável; para extrair o texto é preciso OCR. O normal já guarda o texto."),
    ("qual comando mostra o espaço em disco no Linux?", "Use `df -h` para ver o espaço livre por partição e `du -sh pasta` para o tamanho de uma pasta."),
    ("como escrevo uma boa mensagem de commit?", "Comece com um verbo no imperativo e um resumo curto (até ~50 caracteres), ex.: \"corrige timeout do login\". Se precisar, explique o porquê no corpo."),
    ("o que é uma variável de ambiente?", "É um valor nomeado que o sistema passa para os programas, como `PATH` ou `HOME`. Serve para configurar sem mudar o código."),
    ("pra que serve um índice num banco de dados?", "O índice acelera buscas e ordenações numa coluna, como o índice de um livro. Em troca, ocupa espaço e deixa escritas um pouco mais lentas."),
    ("o que é overfitting?", "É quando o modelo decora os dados de treino e vai mal em dados novos. Dá para perceber quando a perda de validação sobe enquanto a de treino cai."),
    ("o que é um token para um LLM?", "Token é o pedaço de texto que o modelo processa: pode ser uma palavra, parte de uma palavra ou um sinal. Em português, uma palavra costuma virar 1 a 3 tokens."),
    ("qual a diferença entre git merge e git rebase?", "O merge junta duas branches criando um commit de merge; o rebase reaplica seus commits em cima da outra branch, deixando o histórico linear."),
    ("o que é Pix?", "Pix é o sistema de pagamentos instantâneos do Banco Central: transfere dinheiro em segundos, 24h por dia, usando uma chave (CPF, e-mail, telefone ou aleatória)."),
    ("o que é um cron job?", "É uma tarefa agendada para rodar automaticamente em horários definidos, configurada no `crontab` em sistemas Unix."),
    ("como funciona a técnica pomodoro?", "Você trabalha 25 minutos focada, faz 5 de pausa e, a cada 4 ciclos, uma pausa maior de 15 a 30 minutos."),
    ("o que é latência p95?", "É o tempo abaixo do qual ficam 95% das requisições. Mostra a experiência dos casos lentos melhor do que a média."),
]


def _explain(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    question, answer = pick(rng, EXPLANATIONS)
    return {"question": question}, answer


TRANSLATIONS = [
    ("bom dia, tudo bem?", "inglês", "Good morning, how are you?"),
    ("onde fica o banheiro?", "inglês", "Where is the bathroom?"),
    ("a conta, por favor", "espanhol", "La cuenta, por favor."),
    ("estou atrasada para a reunião", "inglês", "I'm late for the meeting."),
    ("quanto custa este livro?", "espanhol", "¿Cuánto cuesta este libro?"),
    ("obrigada pela ajuda", "francês", "Merci pour votre aide."),
    ("pode falar mais devagar?", "inglês", "Could you speak more slowly?"),
    ("o prazo é sexta-feira", "inglês", "The deadline is Friday."),
    ("preciso de um táxi", "espanhol", "Necesito un taxi."),
    ("adorei a apresentação", "inglês", "I loved the presentation."),
]


def _translate(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    text, lang, answer = pick(rng, TRANSLATIONS)
    return {"text": text, "lang": lang}, answer


CREATIVE = [
    ("sugere um título para um post sobre produtividade", "Algumas opções:\n- Menos tarefas, mais resultado\n- O método das 3 prioridades\n- Produtividade sem culpa"),
    ("me dá três ideias de lanche saudável", "- Iogurte natural com frutas e aveia\n- Pão integral com pasta de amendoim\n- Palitos de cenoura com homus"),
    ("escreve uma mensagem curta de parabéns para minha amiga", "Parabéns, amiga! Que seu novo ano venha cheio de saúde, risadas e conquistas. Te adoro!"),
    ("reescreve de forma mais educada: 'manda logo o relatório'", "\"Você consegue me enviar o relatório assim que possível? Obrigada!\""),
    ("faz uma frase motivacional para segunda-feira", "Segunda é página em branco: escreve uma coisa boa nela hoje."),
    ("sugere nomes para um cachorro caramelo", "- Paçoca\n- Doce de Leite\n- Biscoito\n- Mel"),
    ("me ajuda com uma desculpa educada para recusar um convite", "\"Muito obrigada pelo convite! Infelizmente não vou conseguir ir dessa vez, mas espero que seja ótimo.\""),
    ("escreve um poema de quatro versos sobre chuva", "Chove manso na janela,\no café esfria devagar,\na cidade fica mais bela\nquando aprende a desacelerar."),
    ("dá uma dica para dormir melhor", "Tente deixar o celular longe uns 30 minutos antes de deitar e manter o mesmo horário todos os dias."),
    ("me sugere um nome para um bot de finanças", "- Cofrinho\n- Grana Bot\n- Saldo Certo"),
    ("resume em uma frase o que é uma boa reunião", "Uma boa reunião tem objetivo claro, só as pessoas necessárias e termina com responsáveis e prazos definidos."),
    ("me dá ideias de presente até 100 reais", "- Livro de um autor que a pessoa gosta\n- Caneca personalizada\n- Kit de chás\n- Planta pequena com vaso bonito"),
]


def _creative(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    request, answer = pick(rng, CREATIVE)
    return {"request": request}, answer


ABOUT = [
    ("o que você consegue fazer?", "Consigo mexer em arquivos e no terminal, usar git e GitHub, ler PDFs, DOCX e imagens, buscar na web e na sua memória local, criar lembretes e acessar Google, Microsoft 365, Notion, Obsidian e Stripe. É só pedir."),
    ("quem é você?", "Sou o Jarvis, seu assistente pessoal. Rodo na sua máquina e converso com você pelo Telegram."),
    ("você roda na nuvem?", "Não: rodo localmente na sua máquina, com um modelo no Ollama. Só acesso a internet quando uma ferramenta precisa (ex.: busca na web)."),
    ("você guarda minhas conversas?", "Guardo o histórico recente da conversa localmente, num banco SQLite na sua máquina, para manter o contexto."),
]


def _about(rng: random.Random, ctx: Context) -> Tuple[Slots, Any]:
    question, answer = pick(rng, ABOUT)
    return {"question": question}, answer


NO_TOOL_INTENTS: List[Intent] = [
    Intent("chat.greeting", None, [
        "olá!", "bom dia, Jarvis", "boa noite!", "e aí, beleza?", "oi Jarvis, como vai?",
        "fala, Jarvis!", "boa tarde, tudo certo?", "opa, tudo tranquilo?", "oi oi", "bom dia! como você está?",
    ], _reply_pool(GREETING_REPLIES), 1.0),
    Intent("chat.thanks", None, [
        "valeu!", "muito obrigada pela ajuda", "brigada, era isso", "perfeito, obrigada",
        "até amanhã, Jarvis", "tchau!", "show, valeu demais", "ok, era só isso mesmo",
    ], _reply_pool(THANKS_REPLIES), 0.8),
    Intent("chat.percent", None, [
        "calcula {p} por cento de {n}", "{p}% de {n} dá quanto?", "me diz quanto vale {p}% de {n}",
        "se eu tirar {p}% de {n}, quanto é esse desconto?",
    ], _percent, 0.6),
    Intent("chat.multiply", None, [
        "quanto dá {a} vezes {b}?", "multiplica {a} por {b}", "{a} x {b} = ?",
    ], _multiply, 0.5),
    Intent("chat.split", None, [
        "a conta deu R$ {v}, dividindo por {k} pessoas fica quanto?", "divide {v} reais entre {k} amigos",
    ], _split_bill, 0.4),
    Intent("chat.tip", None, [
        "se a conta deu R$ {v}, quanto fica com 10% de serviço?", "calcula a gorjeta de 10% numa conta de {v} reais",
    ], _tip, 0.3),
    Intent("chat.celsius", None, [
        "{c} graus Celsius são quantos Fahrenheit?", "converte {c} °C para Fahrenheit",
    ], _celsius, 0.3),
    Intent("chat.km", None, [
        "{km} km dá quantas milhas?", "converte {km} quilômetros em milhas",
    ], _km_miles, 0.3),
    Intent("chat.explain", None, [
        "{question}", "uma dúvida: {question}", "rapidinho: {question}",
        "sem pesquisar, só me responde: {question}",
    ], _explain, 1.6, tags=["negativo_dificil"]),
    Intent("chat.translate", None, [
        "como se diz '{text}' em {lang}?", "traduz para o {lang}: {text}", "passa pro {lang}: \"{text}\"",
    ], _translate, 0.6),
    Intent("chat.creative", None, [
        "{request}", "rapidinho, {request}", "Jarvis, {request}",
    ], _creative, 1.0),
    Intent("chat.about", None, [
        "{question}", "Jarvis, {question}", "curiosidade: {question}",
    ], _about, 0.4),
]


# ----------------------------------------------- continuação (após a ferramenta)

# (intenção-base, função que inventa a saída da ferramenta e a resposta final)


def _cont_list_dir(rng: random.Random, args: Dict[str, Any]) -> Tuple[str, str]:
    path = args.get("path", ".")
    files = rng.sample(["relatorio.pdf", "notas.md", "foto_01.jpg", "script.py", "dados.json",
                        "README.txt", "backup.tar"], 3)
    folder = rng.choice(["antigos", "2026", "rascunhos"])
    output = "Diretório: {}\n[dir]  {}/\n".format(path, folder) + "\n".join(
        "[file] {} ({} KB)".format(name, rng.randint(2, 900)) for name in files)
    reply = "Em `{}` tem:\n- pasta `{}/`\n".format(path, folder) + "\n".join("- `{}`".format(f) for f in files)
    return output, reply


def _cont_write(rng: random.Random, args: Dict[str, Any]) -> Tuple[str, str]:
    size = len(args["content"].encode("utf-8"))
    return ("Arquivo escrito: {} ({} bytes)".format(args["path"], size),
            "Pronto! Salvei em `{}`.".format(args["path"]))


def _cont_reminder(rng: random.Random, args: Dict[str, Any]) -> Tuple[str, str]:
    rid = rng.randint(2, 60)
    when = args["when"]
    return ("Lembrete #{} agendado para {}: {}".format(rid, when, args["text"]),
            "Combinado: lembrete #{} criado ({}) — {}.".format(rid, when, args["text"]))


def _cont_git_status(rng: random.Random, args: Dict[str, Any]) -> Tuple[str, str]:
    changed = rng.sample(["app/main.py", "README.md", "tests/test_api.py", "docs/guia.md"], 2)
    output = "On branch main\nChanges not staged for commit:\n" + "\n".join(
        "\tmodified:   {}".format(c) for c in changed)
    reply = "Você está na `main` com 2 arquivos modificados e não commitados:\n" + "\n".join(
        "- `{}`".format(c) for c in changed) + "\nPróximo passo: `git add` e commit, se quiser."
    return output, reply


def _cont_find(rng: random.Random, args: Dict[str, Any]) -> Tuple[str, str]:
    found = ["{}/{}".format(rng.choice(["Documentos", "Downloads", "Desktop"]),
                            args["name"].replace("*", rng.choice(["a", "2025", "final"])))]
    return ("\n".join(found), "Achei: `{}`.".format(found[0]))


def _cont_read_file(rng: random.Random, args: Dict[str, Any]) -> Tuple[str, str]:
    path = args["path"]
    output = "Arquivo: {} (linhas 1-3 de 3)\n1: # Título\n2: primeira linha útil\n3: fim".format(path)
    return output, "O `{}` é curto (3 linhas): um título, uma linha de conteúdo e \"fim\".".format(path)


CONTINUATIONS: Dict[str, Callable[[random.Random, Dict[str, Any]], Tuple[str, str]]] = {
    "list_dir.path": _cont_list_dir,
    "write_file.new": _cont_write,
    "reminder.minutes": _cont_reminder,
    "reminder.tomorrow": _cont_reminder,
    "git.status": _cont_git_status,
    "find_files.name": _cont_find,
    "read_file.whole": _cont_read_file,
}


# ------------------------------------------------------- variação de superfície

PREFIXES = ["jarvis, ", "Jarvis, ", "ei, ", "por favor, ", "Jarvis "]
SUFFIXES = [" por favor", " pfv", "!", " valeu", " obrigada"]


def strip_accents(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def vary_surface(rng: random.Random, text: str) -> str:
    """Pequenas variações de escrita (prefixo, sufixo, sem acento, maiúscula).

    Aplicada ao **modelo de frase, antes** de preencher as entidades: assim tirar
    acento ou capitalizar nunca altera um nome de arquivo ou valor de argumento.
    Sem sufixo quando a frase termina numa entidade (ex.: o conteúdo de um
    arquivo), para o sufixo não parecer parte do valor.
    """
    if rng.random() < 0.2:
        text = pick(rng, PREFIXES) + text
    if rng.random() < 0.15 and not text.endswith("?") and not text.rstrip("\"'").endswith("}"):
        text = text + pick(rng, SUFFIXES)
    if rng.random() < 0.12:
        text = strip_accents(text)
    if rng.random() < 0.3:
        text = text[:1].upper() + text[1:]
    return text
