# Prompts Everywhere


from ..meta import MetaField
from common_pos_neg import is_negative_title, is_positive_title
from ..formatters import calc_model_hash, calc_lora_hash, convert_skip_clip, calc_unet_hash

def is_positive_prompt_everywhere(node_id, obj, prompt, extra_data, outputs, input_data_all):
    title = obj['_meta']['title']
    if title:
        if is_positive_title(title):
            return True
    return False

def is_negative_prompt_everywhere(node_id, obj, prompt, extra_data, outputs, input_data_all):
    title = obj['_meta']['title']
    if title:
        if is_negative_title(title):
            return True
    return False

SAMPLERS = {
}

CAPTURE_FIELD_LIST = {
    "ShowText|pysssss": {
        MetaField.POSITIVE_PROMPT: {
            "field_name": "text",
            "validate": is_positive_prompt_everywhere
        },
        MetaField.NEGATIVE_PROMPT: {
            "field_name": "text",
            "validate": is_negative_prompt_everywhere
        },
    }
}