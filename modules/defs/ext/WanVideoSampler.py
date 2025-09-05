from ..meta import MetaField

# Tell the extension which inputs carry the positive/negative conditioning.
# WanVideoSampler packs both prompts into a single "text_embeds" input,
# so we point both to that socket. The extension will trace upstream to
# extract the prompt/negative from the embed creator.
SAMPLERS = {
    "WanVideoSampler": {
        "positive": "text_embeds",
        "negative": "text_embeds",
    },
}

# Tell the extension how to capture the key parameters by their input names.
# Field names reflect WanVideoSampler's INPUTS (steps, cfg, seed, scheduler, denoise_strength, …).
CAPTURE_FIELD_LIST = {
    "WanVideoSampler": {
        MetaField.SEED: {"field_name": "seed"},
        MetaField.STEPS: {"field_name": "steps"},
        MetaField.CFG: {"field_name": "cfg"},
        MetaField.SCHEDULER: {"field_name": "scheduler"},
        # WanVideoSampler uses "denoise_strength" (not "denoise")
        MetaField.DENOISE: {"field_name": "denoise_strength"},
    },
}
