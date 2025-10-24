# Metadata capture definition for the "Model Assembler" custom node.
# This file should be placed within the ComfyUI-Easy-Use node's metadata helper directory.

from ..meta import MetaField
from ..formatters import calc_model_hash, calc_vae_hash

# Note: ComfyUI-Easy-Use does not have a built-in CLIP hash calculator by default.
# We are assuming one could be added to the formatters, similar to the others.
# If it doesn't exist, the 'Clip Model Hash' field will cause an error.
try:
    from ..formatters import calc_clip_hash
except ImportError:
    # Create a dummy function if calc_clip_hash doesn't exist to prevent crashes.
    def calc_clip_hash(name):
        return f"hash_for_{name}"

# --- Custom Selector Functions for Model Assembler ---

def get_assembler_model_name(node_id, obj, prompt, extra_data, outputs, input_data):
    """Determines the primary model name based on the load mode."""
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    if mode == "full_checkpoint":
        return input_data[0].get("ckpt_name", [""])[0]
    else:
        return input_data[0].get("base_model", [""])[0]

def get_assembler_model_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    """Calculates the hash of the primary model."""
    model_name = get_assembler_model_name(node_id, obj, prompt, extra_data, outputs, input_data)
    return calc_model_hash(model_name)

def get_assembler_vae_name(node_id, obj, prompt, extra_data, outputs, input_data):
    """Determines the VAE name based on the load mode."""
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    if mode == "full_checkpoint":
        # In this mode, the VAE is part of the checkpoint.
        return input_data[0].get("ckpt_name", [""])[0]
    else:
        # In separate mode, it's an explicit VAE file.
        return input_data[0].get("vae_model", [""])[0]

def get_assembler_vae_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    """Calculates the hash of the VAE."""
    vae_name = get_assembler_vae_name(node_id, obj, prompt, extra_data, outputs, input_data)
    return calc_vae_hash(vae_name)

def get_assembler_clip_names(node_id, obj, prompt, extra_data, outputs, input_data):
    """Gets a list of all selected, non-empty CLIP model names."""
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    if mode == "full_checkpoint":
        # CLIP is bundled within the checkpoint, no separate names to list.
        return []
    
    clip_names = []
    for key in ["clip_model_1", "clip_model_2", "clip_model_3"]:
        # .get(key, [None])[0] safely gets the value
        name = input_data[0].get(key, [None])[0]
        if name and name != "None":
            clip_names.append(name)
    return clip_names

def get_assembler_clip_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    """Calculates the hashes for all selected CLIP models."""
    names = get_assembler_clip_names(node_id, obj, prompt, extra_data, outputs, input_data)
    return [calc_clip_hash(name) for name in names]


# --- Metadata Capture Definition ---

CAPTURE_FIELD_LIST = {
    "ModelAssembler": {
        # --- Standard MetaFields ---
        MetaField.MODEL_NAME: {"selector": get_assembler_model_name},
        MetaField.MODEL_HASH: {"selector": get_assembler_model_hash},
        MetaField.VAE_NAME: {"selector": get_assembler_vae_name},
        MetaField.VAE_HASH: {"selector": get_assembler_vae_hash},

        # --- Custom Metadata Fields ---
        # These will be saved with string keys in the metadata.
        "Clip Type": {"field_name": "clip_type"},
        "Clip Model Name(s)": {"selector": get_assembler_clip_names},
        "Clip Model Hash(es)": {"selector": get_assembler_clip_hashes},
        "UNet Weight Type": {"field_name": "weight_dtype"},
    }
}