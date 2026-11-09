# Fine-tuning com LoRA (subprojeto 2)

Preparação completa e reprodutível para ajustar um Qwen 2.5 pequeno com LoRA na tarefa que
o subprojeto 1 mede: **escolher a ferramenta certa (ou nenhuma) e preencher os argumentos**.

> **Estado (08/11/2026): treinado localmente e medido.** QLoRA do Qwen2.5-3B-Instruct numa
> RTX 4050 Laptop de 6 GB com [`train_local.py`](train_local.py) (101 passos, 219,6 min, pico
> de 5,08 GB; perda no val 0,917 -> 0,099). No eval de 78 casos, 3 rodadas: base Q8_0 83,3%
> (65/65/65) -> ajustado **100%** (78/78/78), args 81,4% -> 96,7%, McNemar p < 0,001. **Isso
> não prova generalização**: os modelos de frase foram escritos por quem tinha lido o eval
> (ver [`evals/results/FINETUNE.md`](../evals/results/FINETUNE.md)). Os pesos não estão no
> repositório (`finetune/out/` no `.gitignore`); o passo a passo abaixo reproduz.

Especificação: [`docs/superpowers/specs/2026-11-08-finetune-design.md`](../docs/superpowers/specs/2026-11-08-finetune-design.md).
Pendências: seção "Fine-tuning" de [`evals/PENDENCIAS.md`](../evals/PENDENCIAS.md).

## Arquivos

| Arquivo | O que é |
|---|---|
| [`generate_dataset.py`](generate_dataset.py) | gera o dataset a partir do registro e do system prompt reais; filtra vazamento |
| [`templates.py`](templates.py) | intenções, modelos de frase, entidades e respostas sem ferramenta |
| [`leakage.py`](leakage.py) | checagem de vazamento contra `evals/data/tool_selection.jsonl` |
| [`chat_format.py`](chat_format.py) | renderiza a conversa no texto exato que o Ollama monta para o `qwen2.5` |
| [`check_template.py`](check_template.py) | prova a equivalência acima contra o Ollama local |
| [`train_local.py`](train_local.py) | **treino QLoRA local** (6 GB): prompt sem gradiente, perda só na resposta, MLflow |
| [`merge_adapter.py`](merge_adapter.py) | junta o adaptador ao modelo base (CPU, bf16) |
| [`to_gguf.py`](to_gguf.py) | GGUF Q8_0 com o tokenizador do GGUF oficial do `qwen2.5` |
| [`Modelfile.local`](Modelfile.local) | importa o GGUF local no Ollama, template oficial |
| [`train_lora.ipynb`](train_lora.ipynb) | alternativa no Colab (Unsloth, LoRA, GGUF q4_K_M) — **não executado** |
| [`Modelfile`](Modelfile) | importa o GGUF no Ollama com o template oficial do `qwen2.5` |
| `data/train.jsonl`, `data/val.jsonl` | dataset versionado (seed 42) |
| `data/tools.json` | os 25 schemas reais (iguais em todo exemplo; guardados uma vez) |
| `data/stats.json`, `data/leakage_report.json` | composição e relatório de vazamento |

## Passo a passo

Comandos a partir da raiz do repositório (no Linux/macOS, `.venv/bin/python`).

### 1. Gerar os dados (local, segundos, sem GPU)

```bash
.venv/Scripts/python.exe finetune/generate_dataset.py
```

Determinístico: com a mesma seed e o mesmo código sai o mesmo arquivo, byte a byte (o teste
`test_geracao_deterministica_e_valida` e uma regeração conferida com `cmp`). Opções:
`--total` (exemplos pedidos, padrão 2.200), `--no-tool-frac` (0,22), `--continuation-frac`
(0,05), `--val-frac` (fração dos modelos de frase no val, 0,2), `--threshold` (limiar de
vazamento, 0,35), `--seed`.

Paráfrase opcional com um modelo local (leve; só inferência):

```bash
.venv/Scripts/python.exe finetune/generate_dataset.py --paraphrase-frac 0.1 --paraphrase-model qwen2.5:3b --out /tmp/ft
```

Ela fica **desligada no dataset versionado**: num teste com 32 pedidos o `qwen2.5:3b`
aceitou 24 paráfrases, mas algumas vieram com erro de português ("Qualas faturas...",
"Extraí o conteúdo...") ou escaparam para caracteres chineses no meio da frase (agora
descartados). Paráfrase que perde uma entidade (nome de arquivo, valor de argumento) é
descartada; as aceitas passam pelo mesmo filtro de vazamento.

### 2. Checar vazamento (automático)

O gerador já remove os vazados, roda a checagem de novo e **falha** se sobrar algum. O
relatório vai para `data/leakage_report.json` e o teste
`test_dataset_versionado_nao_vaza_o_eval` refaz a checagem no dataset versionado a cada
`pytest`. Detalhes e números na seção [Vazamento](#vazamento).

### 3. Conferir o template (local, ~1 min, só inferência)

```bash
.venv/Scripts/python.exe finetune/check_template.py --model qwen2.5:1.5b
```

Para 6 conversas (pedido com ferramenta, conversa simples, depois do resultado da
ferramenta, histórico com dois argumentos, sem ferramentas, sem system), manda a mesma coisa
pelo `/api/chat` (o Ollama aplica o template) e pelo `/api/generate` com `raw` e o texto do
`chat_format.py`. Resultado em 08/11/2026: **mesmo número de tokens de prompt e mesma saída
nas 6** (descarregando o modelo entre as chamadas para o cache de prompt não interferir).

### 4. Treinar no Colab

1. Empacote a pasta `finetune/` (o notebook precisa de `chat_format.py` e `data/`):
   ```bash
   .venv/Scripts/python.exe -c "import shutil; shutil.make_archive('finetune', 'zip', '.', 'finetune')"
   ```
2. Abra [`train_lora.ipynb`](train_lora.ipynb) no Colab, escolha GPU T4 e rode as células em
   ordem; na célula de dados, envie o `finetune.zip` (ou preencha `REPO_URL`).
3. O notebook mostra a perda no val **antes** do treino e a cada meia época, a contagem real
   de tokens por exemplo (confere que nada é cortado em 4.608) e uma conferência de sanidade
   da ferramenta escolhida numa amostra do val sintético.

Hiperparâmetros (justificativas no notebook): `Qwen2.5-1.5B-Instruct`, LoRA `r=16`,
`lora_alpha=32`, `dropout=0`, atenção + MLP; `lr=2e-4` linear com 5% de warmup; 2 épocas;
batch 2 × acumulação 8 (efetivo 16, ~105 passos por época); `adamw_8bit`; fp16; seed 42;
`max_seq_length=4608`. Perda só na última resposta do assistente. O 3B é uma linha a trocar
(`MODEL_NAME`), com 4 bits.

Custo: cada exemplo tem ~3.900 tokens de prompt (as 25 ferramentas no system), então duas
épocas processam ~13 milhões de tokens. O tempo na T4 **não foi medido**; o notebook grava
`minutos_treino` no `train_report.json`.

### 5. Exportar GGUF q4_K_M

A última parte do notebook junta o LoRA ao modelo base, quantiza em q4_K_M e baixa
`jarvis-qwen-lora.Q4_K_M.gguf`, o adaptador (`.zip`), `train_report.json` e `loss.png`.

### 6. Importar no Ollama

```bash
mv ~/Downloads/jarvis-qwen-lora.Q4_K_M.gguf finetune/     # *.gguf está no .gitignore
ollama create jarvis-qwen-lora -f finetune/Modelfile
.venv/Scripts/python.exe finetune/check_template.py --model jarvis-qwen-lora
```

O `Modelfile` copia **literalmente** o template oficial do `qwen2.5` (o mesmo do treino) e
fixa `num_ctx 8192`, `temperature 0.2` e os `stop` do Qwen. Ele foi validado aqui criando um
modelo temporário com o mesmo arquivo sobre o `qwen2.5:1.5b` (`FROM` trocado; sem baixar
nada): o template ficou idêntico ao oficial e o `check_template.py` passou nas 6 conversas.
O `.gitattributes` força `\n` no `Modelfile`, porque um `\r\n` do autocrlf do Windows
mudaria o prompt.

### 7. Medir o antes/depois com o eval da etapa 1

```bash
.venv/Scripts/python.exe evals/compare_models.py qwen2.5:1.5b jarvis-qwen-lora --skip-rag --markdown evals/results/FINETUNE.md
```

- Mesmos 78 casos, mesmo system prompt real, seed 42, temperatura 0,2, `num_ctx` 8192, modo
  nativo. Rodar a base **na mesma execução** importa: duas rodadas iguais do mesmo modelo
  variam 1–2 casos (no 5b, o `qwen2.5:3b` deu 66/78 e 64/78 com o mesmo prompt), e a
  diferença esperada do LoRA precisa ser comparada com isso.
- Use `--markdown` com outro arquivo: sem ele o `compare_models.py` sobrescreve o
  `evals/results/RESULTS.md` com só dois modelos.
- Se treinar o 3B, compare com `qwen2.5:3b` (e importe com outro nome).
- Os JSON por caso ficam em `evals/results/` e o run no MLflow; guarde `train_report.json` e
  `loss.png` em `finetune/results/`. Só então preencha a linha do modelo ajustado no README
  principal — com o número medido, a categoria "sem ferramenta" e as ressalvas.

## Formato dos dados

Uma conversa por linha, no formato de mensagens do Ollama/OpenAI. A chamada de ferramenta do
assistente segue o que o `/api/chat` do Ollama devolve e aceita de volta no histórico
(`content` vazio, `arguments` como objeto):

```json
{"id": "ft-01484", "kind": "tool", "tool": "reminder", "template": "reminder.minutes.02", "split": "train",
 "messages": [
  {"role": "system", "content": "Você é o Jarvis, assistente pessoal de Julia. ... (encurtado aqui)"},
  {"role": "user", "content": "Em 30 minutos me lembra: renovar o seguro do carro"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
     "function": {"name": "reminder", "arguments": {"action": "add", "when": "30m", "text": "renovar o seguro do carro"}}}]}]}
```

- `kind`: `tool` (pedido -> chamada), `no_tool` (pedido -> resposta em texto) ou
  `continuation` (pedido -> chamada -> resultado com `role: tool` -> resposta final).
- `template`: o modelo de frase; a divisão treino/val é por ele.
- Os schemas ficam em `data/tools.json` (são os mesmos 25 em todo exemplo, na ordem do
  registro); `chat_format.render_example(exemplo, tools)` devolve `(prompt, resposta)`.

**O que o modelo vê e o que ele aprende a gerar** (template do `qwen2.5` no Ollama):

```
<|im_start|>system
Você é o Jarvis ... (system prompt real, com data/hora e pastas)

# Tools

You may call one or more functions to assist with the user query.
...
<tools>
{"type": "function", "function": {reminder Cria, lista e cancela lembretes ... {object <nil> <nil> [action] {"action":{"type":"string","enum":["add","list","delete"]},...}}}}
...
</tools>
...<|im_end|>
<|im_start|>user
Em 30 minutos me lembra: renovar o seguro do carro<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "reminder", "arguments": {"action": "add", "when": "30m", "text": "renovar o seguro do carro"}}
</tool_call><|im_end|>        <- só isto entra na perda
```

### Achado: o Ollama 0.40.14 não manda as ferramentas em JSON para o `qwen2.5`

O template oficial imprime cada ferramenta com `{{ .Function }}`. Nesta versão do Ollama esse
tipo não tem mais um `String()` que devolva JSON, e o Go imprime a struct com `%v`:
`{nome descrição {object <nil> <nil> [obrigatórios] {properties em JSON}}}` — só as
`properties` saem em JSON (com as chaves em ordem alfabética, como todo `map` do Go). Como
foi descoberto: o `apply_chat_template` do Hugging Face daria outro texto, então medi o
prompt real. Um segundo servidor do Ollama com `OLLAMA_DEBUG` (porta separada, sem mexer no
principal) loga o tamanho do prompt em bytes; comparei com o meu renderizador em casos
mínimos até bater byte a byte, pedi ao modelo para copiar o trecho `<tools>` (veio a struct
do Go) e confirmei a ordem das chaves pela saída com temperatura 0.

Consequências:

- É isso que o `qwen2.5` vê **hoje**, inclusive na linha de base do subprojeto 1. O treino
  reproduz o formato em vez de "corrigir", para não criar diferença entre treino e serviço.
- Se o Ollama mudar esse comportamento numa atualização, o prompt muda para a base e para o
  modelo ajustado. Rode `check_template.py` depois de atualizar o Ollama; se falhar, ajuste
  `chat_format.py`, regere os dados e treine de novo.
- Testar um template "corrigido" (ferramentas em JSON) na base é um experimento barato, sem
  treino — está nas pendências.

## Composição do dataset versionado

Seed 42, `--total 2200`: 2.194 exemplos únicos gerados (3 intenções esgotaram as
combinações antes da meta), **60 removidos por vazamento**, **2.134 no final**:

| | Treino | Val | Total |
|---|---|---|---|
| com ferramenta | 1.240 | 307 | 1.547 |
| sem ferramenta | 362 (21,4%) | 117 (26,3%) | 479 (**22,4%**) |
| continuação | 87 | 21 | 108 |
| **total** | **1.689** | **445** | **2.134** |
| modelos de frase | 263 | 71 | 334 |

- **Por ferramenta** (com ferramenta, treino + val): 50 a 65 exemplos para cada uma das 25
  (`google` 50, `stripe` 56, `notion`/`read_docx`/`spreadsheet` 57, `microsoft` 58, as
  demais 62–65). Contagem completa em `data/stats.json`.
- **Sem ferramenta (22,4%)**: saudações, agradecimentos, contas (porcentagem,
  multiplicação, divisão de conta, gorjeta), conversões, traduções, escrita curta,
  perguntas sobre o próprio Jarvis e **99 negativos difíceis** — perguntas que citam git,
  e-mail, PDFs ou comandos mas pedem explicação, não ação ("o que faz o git stash?", "qual
  comando mostra o espaço em disco no Linux?"). Alvo: o `qwen2.5:3b` chamou ferramenta em 3
  de 8 conversas simples. (O `qwen2.5:1.5b` erra no sentido oposto — em 31 dos seus 34
  erros não chama ferramenta —, o que os 72% de exemplos com ferramenta atacam.)
- **Continuação (5%)**: resultado de `list_dir`, `write_file`, `reminder`, `git status`,
  `find_files` ou `read_file` (inventado no formato real da ferramenta) seguido da resposta
  final curta, para o modelo não aprender que toda vez é hora de chamar ferramenta.
- **Variação**: forma de pedir (imperativo, pergunta, "pode...?", com/sem "Jarvis,"),
  entidades sorteadas, datas e pastas de trabalho diferentes no system prompt (POSIX e
  Windows), e em ~12% dos pedidos o texto do modelo de frase vai sem acento (as entidades
  nunca são alteradas). Lembretes e eventos usam a data do próprio
  system prompt ("amanhã às 16h" vira o ISO do dia seguinte).
- Toda chamada é validada contra o JSON Schema real (tipos, `enum`, obrigatórios,
  propriedades conhecidas); um teste garante que `data/tools.json` é igual ao registro atual
  (schema mudou -> regerar).
- 3,6 MB no total, versionado (`finetune/data/` é exceção no `.gitignore` e `-text` no
  `.gitattributes`).

## Vazamento

`evals/data/tool_selection.jsonl` é só de avaliação. Três filtros sobre o texto da usuária
(`leakage.py`, só biblioteca padrão):

| Filtro | Regra | Removidos |
|---|---|---|
| exato | igualdade depois de normalizar (minúsculas, sem acento, sem pontuação) | 0 |
| quase-duplicata | Jaccard de 4-gramas de caracteres ≥ **0,35** | **60** |
| entidade | contém nome de arquivo, URL/domínio, e-mail ou `dono/repo` de algum caso do eval | 0 |

**Como o limiar foi escolhido** (regra fixada antes de gerar): 1,5 × a maior similaridade
entre dois casos **distintos** do próprio eval. Com 4-gramas, o par mais parecido do eval
("lista os arquivos da pasta src" × "acha todos os arquivos PDF na minha pasta Documentos")
tem 0,234, e a mediana do "vizinho mais próximo" de cada caso é 0,116; 1,5 × 0,234 ≈ 0,35.
Para comparação, trocar só a entidade de um caso do eval dá 0,5–0,75 ("roda os testes do
projeto com npm" × "...com pytest" = 0,737). Um teste falha se essa margem sumir.

Sensibilidade (removidos de 2.194): limiar 0,30 → 122 · **0,35 → 60** · 0,40 → 34 ·
0,50 → 7. Dos mantidos, a maior similaridade com o eval tem mediana 0,138, p95 0,286 e
máximo 0,349. Exemplos removidos: "ei, o que tem na minha agenda do Google hoje?" (0,44 com
"o que tenho na minha agenda do Google nos próximos 3 dias?"), "o que está escrito no
carta_recomendacao.docx?" (0,44 com "o que está escrito no arquivo ata_reuniao.docx?").
O mais parecido que passou: "ei, pesquisa na internet: notícias sobre a taxa Selic" (0,349
com "pesquisa na internet as notícias de hoje sobre inteligência artificial").

Zero remoções por entidade porque os pools foram escritos com entidades próprias; o filtro
existe para quem editar os pools (e é testado com entidades do eval).

**Limite honesto:** os modelos de frase foram escritos por quem já tinha lido o eval. Os
filtros pegam cópia e paráfrase próxima, não "ensinar para a prova" no nível da política
(ex.: "anotações -> `knowledge_search`", que também está no system prompt). O teste de
verdade de generalização é um conjunto novo, escrito por outra pessoa. Ver pendências.

## Testes

`tests/test_finetune_format.py` (renderizador e Modelfile), `tests/test_finetune_leakage.py`
(filtros, calibração do limiar, dataset versionado sem vazamento),
`tests/test_finetune_dataset.py` (validação de schema, cobertura das 25 ferramentas,
determinismo, divisão por modelo de frase, `tools.json` = registro, system prompt = o do
`Agent` atual, tamanho) e
`tests/test_finetune_notebook.py` (notebook é JSON válido, sem saídas). Nenhum fala com o
Ollama; `check_template.py` é a conferência que fala.

## Treino local (o que foi feito em 08/11/2026)

Venv separado (Python 3.12; o bot continua no 3.8), fora do git:

```bash
uv venv .venv-train --python 3.12
uv pip install --python .venv-train/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-train/Scripts/python.exe transformers peft bitsandbytes accelerate safetensors gguf "mlflow==2.17.2" "sqlalchemy==2.0.57" "alembic==1.14.1"
```

Versões usadas: torch 2.11.0+cu128, transformers 5.22.2, peft 0.21.2, bitsandbytes 0.50.2,
gguf 0.19.0. O SQLAlchemy fica na 2.0.x porque o MLflow 2.17.2 quebra com a 2.1.

```bash
# descarregue os modelos do Ollama antes (ollama ps / ollama stop): o treino usa ~5 GB de VRAM
.venv-train/Scripts/python.exe finetune/train_local.py --epochs 1 --grad-accum 16 --val-sample 40 --eval-every 25 --max-minutes 210
.venv-train/Scripts/python.exe finetune/merge_adapter.py
.venv-train/Scripts/python.exe finetune/to_gguf.py --base-gguf <FROM de: ollama show qwen2.5:3b --modelfile>
ollama create jarvis-qwen-lora -f finetune/Modelfile.local
# base pelo mesmo caminho, para comparar sem diferença de quantização:
.venv-train/Scripts/python.exe finetune/to_gguf.py --base-gguf <...> --merged <snapshot do Qwen2.5-3B-Instruct no cache do HF> --out finetune/out/qwen2.5-3b-base-q8_0.gguf
.venv/Scripts/python.exe evals/eval_tools.py --model jarvis-qwen-lora --repeats 3
```

Decisões e por quê:

- **Prompt sem gradiente.** O backward completo sobre ~3.900 tokens passou de 7 GB (a GPU tem
  6) e levou 82 s por exemplo, transbordando para a RAM. Com o prompt virando cache de
  chaves/valores sem gradiente, ficou em ~5 GB e 5–7 s. O forward e a perda medida são
  exatos; o gradiente não volta pelas representações do prompt.
- **Sem `prepare_model_for_kbit_training`**: ele converte o embedding (152 mil × 2048) para
  fp32, +1,2 GB.
- **1 época, parada por tempo** em 101 de 106 passos (o lr já estava perto de zero no fim do
  agendamento linear).
- **Q8_0 em vez de q4_K_M**: o Ollama 0.40.14 só quantiza na importação de safetensors (MLX,
  que não aceita Qwen2) e o `gguf` do PyPI não implementa q4_K_M. A base foi convertida igual
  e empata com o `qwen2.5:3b` oficial (p = 1,0), o que confere a conversão.
- O adaptador foi treinado sobre a base em nf4 e juntado à base em bf16 (prática comum do
  QLoRA; a pequena diferença de quantização não foi medida à parte).
