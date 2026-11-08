# Fine-tuning com LoRA do Jarvis — especificação

Data: 2026-11-08 · Branch: `vitrine-ai` · Subprojeto 2 (preparação; o treino fica pendente)

## Objetivo

Deixar pronto, reprodutível e auditável o caminho para ajustar um Qwen 2.5 pequeno com LoRA
na tarefa que o subprojeto 1 mede — **escolher a ferramenta certa (ou nenhuma) e preencher
os argumentos** — e medir o antes/depois com o mesmo `evals/eval_tools.py`.

**O treino não roda nesta máquina** (pedido explícito: não sobrecarregar o notebook, não
baixar pesos do Hugging Face, não instalar torch/transformers/unsloth). Ele fica para o
Google Colab gratuito (T4). Nenhum número do modelo ajustado é escrito antes de existir.

Linha de base já medida (subprojeto 1, `evals/results/RESULTS.md`, modo nativo):

| Modelo | Ferramenta | Args | Sem ferramenta |
|---|---|---|---|
| `qwen2.5:1.5b` | 56,4% (44/78) | 42,6% (26/61) | 100,0% (8/8) |
| `qwen2.5:3b` | 84,6% (66/78) | 82,0% (50/61) | 62,5% (5/8) |

O 1,5B é o padrão do treino (cabe folgado na T4, licença Apache 2.0, mais espaço para
ganhar); o 3B fica opcional (licença Qwen Research, não comercial).

## Peças

| Arquivo | Papel |
|---|---|
| `finetune/chat_format.py` | renderiza a conversa **exatamente** como o template do `qwen2.5` no Ollama (stdlib; usado pelo gerador, pelos testes e pelo notebook) |
| `finetune/check_template.py` | confere o renderizador contra o Ollama local: mesmo nº de tokens de prompt e mesma saída (temperatura 0) entre `/api/chat` e `/api/generate` com `raw` |
| `finetune/templates.py` | modelos de frase por ferramenta/intenção, pools de entidades e respostas sem ferramenta |
| `finetune/leakage.py` | checagem de vazamento contra `evals/data/tool_selection.jsonl` |
| `finetune/generate_dataset.py` | gera `finetune/data/train.jsonl` e `val.jsonl` (seed fixo; paráfrase opcional via Ollama) |
| `finetune/train_lora.ipynb` | Colab T4 + Unsloth: LoRA, perda só nas respostas do assistente, avaliação no val, GGUF q4_K_M |
| `finetune/Modelfile` | importa o GGUF no Ollama com o template oficial do `qwen2.5` |
| `finetune/README.md` | passo a passo de ponta a ponta |

## Decisões

1. **Sem descompasso treino/serviço.** O modelo vai rodar atrás do Ollama, que monta o
   prompt com o template Go do `qwen2.5` (ferramentas no system dentro de `<tools>`,
   chamadas em `<tool_call>`, resultados como `user` com `<tool_response>`). Em vez do
   `apply_chat_template` do Hugging Face (texto diferente: tem um cabeçalho `# Tools`,
   outra serialização dos schemas), o texto de treino é gerado por `chat_format.py`, que
   reproduz o template do Ollama. `check_template.py` prova a equivalência no Ollama local.
2. **System prompt e schemas reais.** O gerador monta o registro real (`build_registry`) e o
   `Agent._system_prompt` real, só trocando a data/hora (sorteada com seed) e a pasta de
   trabalho (um pool de caminhos POSIX e Windows). Se uma ferramenta nova entrar no registro
   sem modelos de frase, o gerador falha. Todo argumento gerado é validado contra o JSON
   Schema da ferramenta (tipos, `enum`, `required`, propriedades conhecidas).
3. **Composição.** ~25 ferramentas × várias intenções × várias formas de pedir; **~20% sem
   ferramenta** (o eval tem ~10%; é o ponto fraco medido do 3B), incluindo "negativos
   difíceis" — perguntas que citam git, Python, e-mail ou agenda, mas pedem explicação, não
   ação. Uma fração pequena (~5%) de exemplos de **continuação** (resultado de ferramenta →
   resposta final em texto), para o modelo não aprender a chamar ferramenta para sempre.
4. **Divisão treino/val por modelo de frase**, não por exemplo: as frases do val nunca
   aparecem no treino, então a perda do val mede generalização para formas novas de pedir.
5. **Vazamento** contra os 78 casos do eval, três filtros (exemplo removido se qualquer um
   disparar):
   - duplicata exata do pedido normalizado (minúsculas, sem acento, sem pontuação, espaços
     colapsados);
   - quase-duplicata: Jaccard de 4-gramas de caracteres do pedido normalizado ≥ limiar
     (calibrado olhando a distribuição: as maiores similaridades *entre casos distintos do
     próprio eval*; valor e justificativa no README);
   - entidade rara do eval (nome de arquivo, URL, e-mail, `dono/repo`) no pedido.
   O gerador imprime o relatório e grava `finetune/data/leakage_report.json`; depois do
   filtro, a checagem roda de novo e **falha** se sobrar algo.
6. **Honestidade sobre o viés de quem escreveu.** Os modelos de frase foram escritos por
   quem já tinha lido o eval. Os filtros pegam cópia e paráfrase próxima, não "ensinar para
   a prova" no nível da política (ex.: "anotações → `knowledge_search`", que também está no
   system prompt). Isso fica registrado em `evals/PENDENCIAS.md`; o teste de verdade é um
   conjunto novo escrito por outra pessoa.

## Formato do JSONL

Uma conversa por linha, no formato de mensagens do Ollama/OpenAI. Os 25 schemas são os
mesmos em todo exemplo, então ficam uma vez só em `finetune/data/tools.json` (com eles em
cada linha o dataset passaria de 20 MB):

```json
{"id": "ft-01484", "kind": "tool", "tool": "reminder", "template": "reminder.minutes.02", "split": "train",
 "messages": [{"role": "system", "content": "Você é o Jarvis..."},
              {"role": "user", "content": "Em 30 minutos me lembra: renovar o seguro do carro"},
              {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                "function": {"name": "reminder", "arguments": {"action": "add", "when": "30m", "text": "renovar o seguro do carro"}}}]}]}
```

`kind` ∈ `tool`, `no_tool`, `continuation`. A perda é calculada só sobre a **última**
mensagem do assistente (e o `<|im_end|>` que a fecha).

**Revisão durante a implementação:** a decisão 1 se mostrou mais importante do que parecia.
No Ollama 0.40.14 o `{{ .Function }}` do template oficial imprime a struct do Go (`%v`), não
JSON — o `apply_chat_template` do Hugging Face teria treinado o modelo num formato de
ferramentas que ele nunca veria no Ollama. Detalhes em `finetune/README.md`.

## Avaliação depois do treino

Importar o GGUF com o `Modelfile` como `jarvis-qwen-lora` e rodar
`evals/compare_models.py qwen2.5:1.5b jarvis-qwen-lora --skip-rag` (mesmo dataset, seed 42,
temperatura 0,2, `num_ctx` 8192, system prompt real). A tabela do README ganha a linha do
modelo ajustado só com o número medido.

## Fora de escopo

Treinar aqui; conversas de vários turnos no eval; DPO/RL; mesclar o LoRA no 3B por padrão.
