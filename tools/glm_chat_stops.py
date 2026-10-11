"""GLM assistant-turn stops, matching the existing serving policy."""


def chat_stop_ids(metadata):
    stops = {int(metadata[key]) for key in ('tokenizer.ggml.eos_token_id',
             'tokenizer.ggml.eot_token_id', 'tokenizer.ggml.eom_token_id') if key in metadata}
    stops.update(index for index, token in enumerate(metadata.get('tokenizer.ggml.tokens', []))
                 if token in ('<|user|>', '<|observation|>'))
    return sorted(stops)
