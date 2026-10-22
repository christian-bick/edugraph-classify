"""Reusable JSON token constraints without the legacy Transformers integration."""
from __future__ import annotations

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def tokenizer_constraints(tokenizer):
    # Decode in context so tokens that start words retain their leading space.
    # The pinned Qwen TokenizersBackend is used directly, without rebuilding it.
    anchor = tokenizer.encode("0", add_special_tokens=False)
    prefix = tokenizer.decode(anchor, clean_up_tokenization_spaces=False)
    special = set(tokenizer.all_special_ids)
    regular = []
    for token in range(len(tokenizer)):
        if token not in special:
            contextual = tokenizer.decode([*anchor, token], clean_up_tokenization_spaces=False)[len(prefix):]
            standalone = tokenizer.decode([token], clean_up_tokenization_spaces=False)
            regular.append((token, contextual, len(contextual) > len(standalone)))
    return TokenEnforcerTokenizerData(regular,
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False).rstrip("\ufffd"),
        tokenizer.eos_token_id, False, len(tokenizer))


def json_constraint(data, schema):
    """One independent parser per generated image; token data is cached per run."""
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
    return lambda batch_id, tokens: enforcer.get_allowed_tokens(tokens.tolist()).allowed_tokens
