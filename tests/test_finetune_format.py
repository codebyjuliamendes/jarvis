"""finetune/chat_format.py: o texto de treino tem de ser o que o Ollama monta.

As strings esperadas foram conferidas contra o Ollama 0.40.14 (mesmo nº de
bytes e de tokens de prompt, e mesma saída com temperatura 0; ver
finetune/check_template.py). Nenhum teste aqui fala com o Ollama.
"""
from __future__ import annotations

from pathlib import Path

from finetune import chat_format as cf

TOOL = {
    "type": "function",
    "function": {
        "name": "t",
        "description": "d <a&b>",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "e"},
                "a": {"type": "array", "items": {"type": "string"}, "description": "f"},
            },
            "required": ["x", "a"],
        },
    },
}

HEADER = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
)


def test_ferramenta_sai_como_struct_do_go_com_properties_em_json_ordenado():
    # nome e descrição soltos, <nil> <nil>, required no formato do Go, properties em JSON
    # com as chaves em ordem alfabética ("a" antes de "x") e sem escapar a descrição.
    expected = (
        '{t d <a&b> {object <nil> <nil> [x a] '
        '{"a":{"type":"array","items":{"type":"string"},"description":"f"},'
        '"x":{"type":"string","description":"e"}}}}'
    )
    assert cf.render_function(TOOL["function"]) == expected


def test_required_vazio_vira_colchetes():
    fn = {"name": "n", "description": "d", "parameters": {"type": "object", "properties": {}}}
    assert cf.render_function(fn) == "{n d {object <nil> <nil> [] {}}}"


def test_prompt_com_ferramentas_igual_ao_template():
    prompt = cf.render_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "oi"}], [TOOL]
    )
    expected = (
        "<|im_start|>system\nS" + HEADER
        + '\n{"type": "function", "function": ' + cf.render_function(TOOL["function"]) + "}"
        + FOOTER + "<|im_start|>user\noi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert prompt == expected


def test_prompt_sem_ferramentas():
    prompt = cf.render_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "oi"}])
    assert prompt == (
        "<|im_start|>system\nS<|im_end|>\n"
        "<|im_start|>user\noi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_sem_system_usa_o_padrao_do_modelfile():
    prompt = cf.render_prompt([{"role": "user", "content": "oi"}])
    assert prompt.startswith("<|im_start|>system\n" + cf.DEFAULT_SYSTEM + "<|im_end|>\n")


def test_historico_com_chamada_e_resultado():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "t", "arguments": {"x": "1 <b>", "a": ["ção"]}}}]},
        {"role": "tool", "content": "ok"},
    ]
    prompt = cf.render_prompt(messages)
    # argumentos do histórico: JSON do Go (compacto, chaves ordenadas, < e > escapados)
    assert (
        '<|im_start|>assistant\n<tool_call>\n{"name": "t", "arguments": '
        '{"a":["ção"],"x":"1 \\u003cb\\u003e"}}\n</tool_call><|im_end|>\n' in prompt
    )
    assert prompt.endswith(
        "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )


def test_papeis_repetidos_sao_fundidos_como_no_ollama():
    prompt = cf.render_prompt([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ])
    assert "<|im_start|>user\na\n\nb<|im_end|>\n" in prompt


def test_render_example_separa_prompt_e_resposta():
    example = {"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "t", "arguments": {"x": "1", "a": ["b"]}}}]},
    ]}
    prompt, completion = cf.render_example(example, [TOOL])
    assert prompt.endswith("<|im_start|>assistant\n")
    # resposta: estilo do Qwen (json.dumps com espaços, ordem do schema) + token de parada
    assert completion == (
        '<tool_call>\n{"name": "t", "arguments": {"x": "1", "a": ["b"]}}\n</tool_call><|im_end|>'
    )


def test_render_example_texto():
    example = {"messages": [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "Olá!"}]}
    assert cf.render_example(example)[1] == "Olá!<|im_end|>"


def test_render_example_exige_assistente_no_fim():
    import pytest

    with pytest.raises(ValueError):
        cf.render_example({"messages": [{"role": "user", "content": "oi"}]})


def test_modelfile_tem_o_template_oficial():
    text = (Path(__file__).resolve().parent.parent / "finetune" / "Modelfile").read_text(encoding="utf-8")
    assert "FROM ./jarvis-qwen-lora.Q4_K_M.gguf" in text
    template = text.split('TEMPLATE """', 1)[1].split('"""', 1)[0]
    # os trechos fixos que o renderizador reproduz
    assert "\n\n# Tools\n\nYou may call one or more functions" in template
    assert '{"type": "function", "function": {{ .Function }}}' in template
    assert '{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}' in template
    assert "<tool_response>\n{{ .Content }}\n</tool_response>" in template
    assert "SYSTEM " + cf.DEFAULT_SYSTEM in text
