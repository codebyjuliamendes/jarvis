# Experimentos sem treino

## Ferramentas em JSON no template do `qwen2.5:3b` (08/11/2026)

**Pergunta.** O template oficial do `qwen2.5` no Ollama 0.40.14 imprime cada ferramenta com
`{{ .Function }}`, que sai como struct do Go (`{nome descrição {object <nil> <nil> [obrigatórios]
{properties}}}`), e não como o JSON que o Qwen viu no treino. Mostrar as ferramentas em JSON
melhora a escolha de ferramenta, sem treino nenhum?

**O que mudou.** Só a linha da ferramenta em
[`Modelfile.json-tools`](Modelfile.json-tools): `{{ .Function }}` → `{{ json .Function }}`. A
função `json` existe no pacote de templates do Ollama (o `ollama create` aceitou; uma função
inexistente falha no parse). Mesmo blob de pesos, mesmo SYSTEM, sem `PARAMETER` (o oficial não
tem nenhum).

**O template entrega o que se espera?** Sim, conferido por
[`check_json_tools.py`](check_json_tools.py) (método do `finetune/check_template.py`): a mesma
conversa pelo `/api/chat` do modelo experimental e pelo `/api/generate` cru com o texto
renderizado em Python (`json.Marshal` do Go: compacto, campos na ordem da struct, `properties`
em ordem alfabética, `required` omitido quando vazio) deu o **mesmo número de tokens e a mesma
saída** nas 2 conversas. O prompt com as 25 ferramentas cresce de ~3.928 para ~4.061 tokens
(+133, +3,4%).

**Resultado (3 rodadas, nativo, mesmo eval de 78 casos, mesma noite):**

| Modelo | Acertos por rodada | Média | Desvio | IC 95% (bootstrap) | Args (média) | Latência p50 |
|---|---|---|---|---|---|---|
| `qwen2.5:3b` (template oficial) | 67 / 67 / 66 | 85,5% | 0,7 p.p. | [77,8; 92,3%] | 84,7% | 0,60 s |
| `jarvis-qwen-json-tools` | 66 / 67 / 68 | 85,9% | 1,3 p.p. | [78,2; 92,7%] | 82,5% | 0,63 s |

Pareado por caso (acerto pela maioria das rodadas): 3 casos só o oficial acerta, 3 só o
experimental; McNemar exato p = 1,000; diferença −0,4 ponto, IC 95% [−6,4; +5,1]. Tabela gerada
em [`evals/results/JSON_TOOLS.md`](../../evals/results/JSON_TOOLS.md).

**Conclusão: resultado nulo.** Trocar a struct do Go por JSON não mudou a escolha de
ferramenta de forma mensurável com 78 casos (o IC da diferença cabe uma melhora ou piora de
~5 pontos). Argumentos ficaram 2 pontos abaixo, dentro da variação entre rodadas do
experimental (±2,5). O custo é pequeno (+133 tokens, ~+0,03 s de mediana). **Não adotado:** o
template oficial continua, e o `finetune/Modelfile` segue reproduzindo o formato que o Ollama
realmente usa. Leitura provável: o modelo já extrai nome, descrição e as `properties` (que já
eram JSON na struct) — o que muda é só o envelope.

**Limpeza.** O modelo temporário foi removido (`ollama rm jarvis-qwen-json-tools`). Os pesos
não foram copiados (o `create` reaproveita o blob do `qwen2.5:3b`).

Reproduzir:

```bash
ollama create jarvis-qwen-json-tools -f finetune/experimentos/Modelfile.json-tools
.venv/Scripts/python.exe finetune/experimentos/check_json_tools.py
.venv/Scripts/python.exe evals/eval_tools.py --model jarvis-qwen-json-tools --mode native --repeats 3
ollama rm jarvis-qwen-json-tools
```
