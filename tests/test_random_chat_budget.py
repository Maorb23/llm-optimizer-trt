"""Exercise prompt fitting without importing GPU/HTTP benchmark dependencies."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def fit_prompt():
    source = Path(__file__).parents[1] / "src/llm_optimizer/bench_client.py"
    tree = ast.parse(source.read_text())
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fixed_length_chat_prompt"
    )
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[helper.name]


class BoundaryTokenizer:
    """Content boundaries skip odd budgets; a padding word fills one token."""

    def encode(self, text, add_special_tokens=True):
        if not add_special_tokens:
            return list(text)
        assert text.startswith("<user>")
        assert text.endswith("</user><assistant>")
        content = text[len("<user>"):-len("</user><assistant>")]
        padding = content.count(" x")
        return list(range(4 + 2 * len(content.replace(" x", "")) + padding))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<user>" + messages[0]["content"] + "</user><assistant>"


@pytest.mark.parametrize("target", [15, 32, 511, 512])
@pytest.mark.parametrize("content", ["abc", "abcdef" * 200])
def test_boundary_changes_preserve_exact_budget_and_template(fit_prompt, target, content):
    tokenizer = BoundaryTokenizer()
    prompt = fit_prompt(tokenizer, content, target)
    assert len(tokenizer.encode(prompt)) == target
    assert prompt.endswith("</user><assistant>")
    assert fit_prompt(tokenizer, content, target) == prompt
    assert fit_prompt(tokenizer, content, target, False) == tokenizer.encode(prompt)


def test_template_exceeding_budget_fails_explicitly(fit_prompt):
    with pytest.raises(ValueError, match="exactly 2"):
        fit_prompt(BoundaryTokenizer(), "abc", 2)
