# WanVideoLoraSelectMulti

from ..meta import MetaField
from ..formatters import calc_lora_hash

def _coerce_to_scalar_strength(val, default=1.0):
    """Coerce strength-like values to a single float scalar."""
    if val is None:
        return default
    # If it's a list/tuple, prefer first numeric element
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return default
        candidate = val[0]
        return _coerce_to_scalar_strength(candidate, default)
    # Strings like "1.0" -> float
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return default
    # Numeric types
    try:
        return float(val)
    except Exception:
        return default

def _coerce_to_string_name(val):
    """Coerce different possible name/path shapes to a string (or None)."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return None
        return _coerce_to_string_name(val[0])
    if isinstance(val, str):
        return val
    # dicts with 'path' or 'name'
    if isinstance(val, dict):
        return val.get("path") or val.get("name") or val.get("model") or None
    try:
        return str(val)
    except Exception:
        return None

def _extract_prev_lora_list(item):
    """
    Normalize prev_lora / lora_stack items into [(name_or_path, strength, clip), ...]
    """
    results = []
    if item is None:
        return results

    if isinstance(item, (list, tuple)):
        for el in item:
            if el is None:
                continue
            if isinstance(el, dict):
                name_raw = el.get("path") or el.get("name") or el.get("model") or el.get("model_name")
                name = _coerce_to_string_name(name_raw)
                strength = _coerce_to_scalar_strength(el.get("strength", 1.0), default=1.0)
                clip = el.get("clip_strength") or el.get("clip") or el.get("clip_scale") or None
                clip = _coerce_to_scalar_strength(clip, default=None) if clip is not None else None
                if name:
                    results.append((name, strength, clip))
            elif isinstance(el, (list, tuple)):
                # (name, strength, clip?)
                name = _coerce_to_string_name(el[0]) if len(el) > 0 else None
                strength = _coerce_to_scalar_strength(el[1], default=1.0) if len(el) > 1 else 1.0
                clip = _coerce_to_scalar_strength(el[2], default=None) if len(el) > 2 else None
                if name:
                    results.append((name, strength, clip))
            elif isinstance(el, str):
                results.append((el, 1.0, None))
    elif isinstance(item, dict):
        # maybe nested structure containing keys we want
        for k in ("prev_lora", "lora", "loras", "lora_stack", "lora_list"):
            if k in item and item[k]:
                results.extend(_extract_prev_lora_list(item[k]))
    return results

def get_wan_lora_stack_from_inputs(input_data):
    """
    Returns list of (model_name_or_path, strength, clip_strength_or_None)
    - First prefer prev_lora / lora_stack upstream values.
    - If none found, fall back to reading lora_0..lora_4 + strength_0..strength_4.
    """
    results = []

    # First scan upstream input_data for prev_lora / lora_stack
    for item in input_data:
        if not item:
            continue
        if isinstance(item, dict):
            # check common keys
            for key in ("prev_lora", "lora", "loras", "lora_stack", "lora_list"):
                if key in item and item[key]:
                    results.extend(_extract_prev_lora_list(item[key]))
            # Also handle nested 'lora_stack' like some nodes produce
            if "lora_stack" in item and item["lora_stack"]:
                for stack in item["lora_stack"]:
                    results.extend(_extract_prev_lora_list(stack))

    if results:
        # filter out invalid entries and ensure scalar strengths
        filtered = []
        for (n, s, c) in results:
            name = _coerce_to_string_name(n)
            if not name:
                continue
            strength = _coerce_to_scalar_strength(s, default=1.0)
            clip = _coerce_to_scalar_strength(c, default=None) if c is not None else None
            filtered.append((name, strength, clip))
        return filtered

    # fallback: merge input_data dicts and read lora_0..4
    merged = {}
    for item in input_data:
        if isinstance(item, dict):
            merged.update(item)

    for i in range(5):
        lkey = f"lora_{i}"
        skey = f"strength_{i}"
        if lkey in merged:
            lval = merged.get(lkey)
            # skip "none" or empty
            if (isinstance(lval, str) and lval.lower() == "none") or not lval:
                continue
            sval = merged.get(skey, 1.0)
            strength = _coerce_to_scalar_strength(sval, default=1.0)
            if strength == 0.0:
                continue
            name = _coerce_to_string_name(lval)
            if not name:
                continue
            results.append((name, strength, None))
    return results

def get_wan_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data):
    stack = get_wan_lora_stack_from_inputs(input_data)
    return [entry[0] for entry in stack]

def get_wan_lora_model_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    names = get_wan_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data)
    hashes = []
    for n in names:
        try:
            h = calc_lora_hash(n, input_data)
        except Exception:
            h = None
        hashes.append(h)
    return hashes

def get_wan_lora_strength_model(node_id, obj, prompt, extra_data, outputs, input_data):
    stack = get_wan_lora_stack_from_inputs(input_data)
    return [entry[1] for entry in stack]

def get_wan_lora_strength_clip(node_id, obj, prompt, extra_data, outputs, input_data):
    stack = get_wan_lora_stack_from_inputs(input_data)
    return [entry[2] for entry in stack]

CAPTURE_FIELD_LIST = {
    "WanVideoLoraSelectMulti": {
        MetaField.LORA_MODEL_NAME: {"selector": get_wan_lora_model_names},
        MetaField.LORA_MODEL_HASH: {"selector": get_wan_lora_model_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_wan_lora_strength_model},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_wan_lora_strength_clip},
    },
}