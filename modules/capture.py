import json
import os
from collections import defaultdict
from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField
from .defs.formatters import calc_lora_hash, calc_model_hash, extract_embedding_names, extract_embedding_hashes

from nodes import NODE_CLASS_MAPPINGS
from .trace import Trace
from execution import get_input_data
from comfy_execution.graph import DynamicPrompt


class Capture:
    @classmethod
    def get_inputs(cls):
        inputs = {}
        prompt = hook.current_prompt
        extra_data = hook.current_extra_data
        outputs = hook.prompt_executer.caches.outputs

        for node_id, obj in prompt.items():
            class_type = obj["class_type"]
            obj_class = NODE_CLASS_MAPPINGS[class_type]
            node_inputs = prompt[node_id]["inputs"]
            input_data = get_input_data(
                node_inputs, obj_class, node_id, outputs, DynamicPrompt(prompt), extra_data
            )

            # Process field data mappings for the captured inputs
            for node_class, metas in CAPTURE_FIELD_LIST.items():
                if class_type != node_class:
                    continue
                
                for meta, field_data in metas.items():
                    # Skip invalidated nodes
                    if field_data.get("validate") and not field_data["validate"](
                        node_id, obj, prompt, extra_data, outputs, input_data
                    ):
                        continue

                    # Initialize list for meta if not exists
                    if meta not in inputs:
                        inputs[meta] = []

                    # Get field value or selector
                    value = field_data.get("value")
                    if value is not None:
                        inputs[meta].append((node_id, value))
                    else:
                        selector = field_data.get("selector")
                        if selector:
                            v = selector(node_id, obj, prompt, extra_data, outputs, input_data)
                            cls._append_value(inputs, meta, node_id, v)
                            continue

                        # Fetch and process value from field_name
                        field_name = field_data["field_name"]
                        value = input_data[0].get(field_name)
                        if value is not None:
                            format_func = field_data.get("format")
                            v = cls._apply_formatting(value, input_data, format_func)
                            cls._append_value(inputs, meta, node_id, v)

        return inputs

    @staticmethod
    def _apply_formatting(value, input_data, format_func):
        """Apply formatting to a value using the given format function."""
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        if format_func:
            value = format_func(value, input_data)
        return value

    @staticmethod
    def _append_value(inputs, meta, node_id, value):
        """Append processed value to the inputs list."""
        if isinstance(value, list):
            for x in value:
                inputs[meta].append((node_id, x))
        elif value is not None:
            inputs[meta].append((node_id, value))

    @classmethod
    def get_lora_strings_and_hashes(cls, inputs_before_sampler_node):
        lora_names = inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, [])
        lora_weights = inputs_before_sampler_node.get(MetaField.LORA_STRENGTH_MODEL, [])
        lora_hashes = inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, [])

        lora_strings = []
        lora_hashes_list = []

        for name, weight, hash_val in zip(lora_names, lora_weights, lora_hashes):
            if not (name and weight and hash_val):
                continue

            clean_name = os.path.splitext(os.path.basename(name[1]))[0].replace(' ', '_').replace(':', '_')

            # LoRA strings for prompt and "Hashes" list
            lora_strings.append(f"<lora:{clean_name}:{weight[1]}>")
            lora_hashes_list.append(f"{clean_name}: {hash_val[1]}")

        lora_hashes_string = ", ".join(lora_hashes_list)
        return lora_strings, lora_hashes_string

    @classmethod
    def gen_pnginfo_dict(cls, inputs_before_sampler_node, inputs_before_this_node, prompt, save_civitai_sampler=True):
        pnginfo_dict = {}

        # Collect all metadata if sampler node missing
        if not inputs_before_sampler_node:
            inputs_before_sampler_node = defaultdict(list)
            cls._collect_all_metadata(prompt, inputs_before_sampler_node)

        # Helper function to update PNG info with a single field
        def update_field(inputs, metafield, key):
            # Get the list for the field
            values = inputs.get(metafield, [])
            
            # Check if the list is not empty and the first item has a second element
            if values and len(values[0]) > 1:
                value = values[0][1]
            else:
                value = None
            
            # If value is not None or empty, update the pnginfo_dict
            if value not in [None, ""]:
                pnginfo_dict[key] = value
            return value  # Return the value (or None)

        # Update main fields
        positive_prompt = update_field(inputs_before_sampler_node, MetaField.POSITIVE_PROMPT, "Positive prompt")
        if positive_prompt is None:
            positive_prompt = ""
            print("[ComfyUI Image Metadata Extension] WARNING: Positive prompt is empty!")

        negative_prompt = update_field(inputs_before_sampler_node, MetaField.NEGATIVE_PROMPT, "Negative prompt")
        if negative_prompt is None:
            negative_prompt = ""
            print("[ComfyUI Image Metadata Extension] WARNING: Negative prompt is empty!")

        lora_strings, lora_hashes_string = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)

        # Append Lora models to the positive prompt, which is required for the Civitai website to parse and apply Lora weights.
        # Format: <lora:Lora_Model_Name:weight_value>. Example: <lora:Lora_Name_00:0.6> <lora:Lora_Name_01:0.8>
        if lora_strings:
            positive_prompt += " " + " ".join(lora_strings)

        pnginfo_dict["Positive prompt"] = positive_prompt.strip()
        pnginfo_dict["Negative prompt"] = negative_prompt.strip()
        
        update_field(inputs_before_sampler_node, MetaField.STEPS, "Steps")

        # Sampler and Scheduler handling
        sampler_names = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME, [])
        schedulers = inputs_before_sampler_node.get(MetaField.SCHEDULER, [])
        if save_civitai_sampler:
            pnginfo_dict["Sampler"] = cls.get_sampler_for_civitai(sampler_names, schedulers)
        else:
            if sampler_names:
                pnginfo_dict["Sampler"] = sampler_names[0][1]
                if schedulers and schedulers[0][1] != "normal":
                    pnginfo_dict["Sampler"] += f"_{schedulers[0][1]}"

        # Additional fields
        update_field(inputs_before_sampler_node, MetaField.CFG, "CFG scale")
        update_field(inputs_before_sampler_node, MetaField.SEED, "Seed")
        update_field(inputs_before_sampler_node, MetaField.CLIP_SKIP, "Clip skip")

        # Image size
        image_width = inputs_before_sampler_node.get(MetaField.IMAGE_WIDTH, [[None]])[0][1]
        image_height = inputs_before_sampler_node.get(MetaField.IMAGE_HEIGHT, [[None]])[0][1]
        if isinstance(image_width, int) and isinstance(image_height, int) and image_width > 0 and image_height > 0:
            pnginfo_dict["Size"] = f"{image_width}x{image_height}"

        update_field(inputs_before_sampler_node, MetaField.MODEL_NAME, "Model")
        update_field(inputs_before_sampler_node, MetaField.MODEL_HASH, "Model hash")
        update_field(inputs_before_this_node, MetaField.VAE_NAME, "VAE")
        update_field(inputs_before_this_node, MetaField.VAE_HASH, "VAE hash")

        # Handle Denoising Strength
        denoise_value = inputs_before_sampler_node.get(MetaField.DENOISE, [])
        if denoise_value:
            denoise_value = denoise_value[0][1] if isinstance(denoise_value[0], (tuple, list)) else denoise_value[0]
            if 0 < float(denoise_value) < 1:
                pnginfo_dict["Denoising strength"] = float(denoise_value)

        # Check for 'Hires upscale' or 'Hires upscaler'
        hires_upscale = inputs_before_this_node.get(MetaField.UPSCALE_BY, [])
        hires_upscaler = inputs_before_this_node.get(MetaField.UPSCALE_MODEL_NAME, [])
        if hires_upscale or hires_upscaler:
            pnginfo_dict["Denoising strength"] = denoise_value or 1.0 # if 'Hires upscale' or 'Hires upscaler' always add Denoise field

        # Add Hi-Res, based on https://github.com/civitai/civitai/blob/0c6a61b2d3ee341e77a357d4c08cf220e22b1190/src/server/common/model-helpers.ts#L33
        update_field(inputs_before_this_node, MetaField.UPSCALE_BY, "Hires upscale")
        update_field(inputs_before_this_node, MetaField.UPSCALE_MODEL_NAME, "Hires upscaler")

        # Add Lora hashes, based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/extensions-builtin/Lora/scripts/lora_script.py#L78
        if lora_hashes_string:
            pnginfo_dict["Lora hashes"] = f'"{lora_hashes_string}"'

        # Update with Lora and Embeddings
        pnginfo_dict.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo_dict.update(cls.gen_embeddings(inputs_before_sampler_node))

        # Add model hashes
        hashes_for_civitai = cls.get_hashes_for_civitai(inputs_before_sampler_node, inputs_before_this_node)
        if hashes_for_civitai:
            pnginfo_dict["Hashes"] = json.dumps(hashes_for_civitai)

        return pnginfo_dict

    @classmethod
    def _collect_all_metadata(cls, prompt, result_dict):
        node_fields = {
            "prompt": {"positive", "negative"},
            "denoise": {"denoise"},
            "sampler": {"seed", "steps", "cfg", "sampler_name", "scheduler"},
            "size": {"width", "height"},
            "model": {"ckpt_name"},
        }
        resolved_nodes = {
            name: Trace.find_node_with_fields(prompt, fields)
            for name, fields in node_fields.items()
        }
        cls._collect_lora_metadata(prompt, result_dict)
        cls._collect_model_metadata(resolved_nodes.get("model"), result_dict)
        cls._collect_scalar_fields(resolved_nodes.get("denoise"), MetaField.DENOISE, "denoise", result_dict)
        cls._collect_multiple_fields(resolved_nodes.get("sampler"), result_dict, {
            "sampler_name": MetaField.SAMPLER_NAME,
            "scheduler": MetaField.SCHEDULER,
            "seed": MetaField.SEED,
            "steps": MetaField.STEPS,
            "cfg": MetaField.CFG,
        })
        cls._collect_multiple_fields(resolved_nodes.get("size"), result_dict, {
            "width": MetaField.IMAGE_WIDTH,
            "height": MetaField.IMAGE_HEIGHT,
        })
        # TODO: Add embedding metadata collection
        cls._collect_prompt_metadata(resolved_nodes.get("prompt"), prompt, result_dict)

    @classmethod
    def _collect_lora_metadata(cls, prompt, result_dict):
        for lora_node_id, lora_node in Trace.find_all_nodes_with_fields(prompt, {"lora_name", "strength_model"}):
            inputs = lora_node.get("inputs", {})
            name = inputs.get("lora_name")
            strength = inputs.get("strength_model")
            if name:
                result_dict[MetaField.LORA_MODEL_NAME].append((lora_node_id, name, 0))
                result_dict[MetaField.LORA_MODEL_HASH].append((lora_node_id, calc_lora_hash(name), 0))
            if strength:
                result_dict[MetaField.LORA_STRENGTH_MODEL].append((lora_node_id, strength, 0))

    @classmethod
    def _collect_model_metadata(cls, model_data, result_dict):
        if not model_data:
            return
        model_id, model_node = model_data
        if model_node:
            name = model_node["inputs"].get("ckpt_name")
            if name:
                result_dict[MetaField.MODEL_NAME].append((model_id, name, 0))
                result_dict[MetaField.MODEL_HASH].append((model_id, calc_model_hash(name), 0))

    @classmethod
    def _collect_scalar_fields(cls, data, meta_key, field_key, result_dict):
        if not data:
            return
        node_id, node = data
        if node:
            value = node["inputs"].get(field_key)
            if value is not None:
                result_dict[meta_key].append((node_id, value, 0))

    @classmethod
    def _collect_multiple_fields(cls, data, result_dict, field_map):
        if not data:
            return
        node_id, node = data
        if node:
            for field, meta in field_map.items():
                val = node["inputs"].get(field)
                if val is not None:
                    result_dict[meta].append((node_id, val, 0))

    @classmethod
    def _collect_prompt_metadata(cls, data, prompt, result_dict):
        if not data:
            return
        prompt_id, prompt_node = data
        if prompt_node:
            for label, meta in {"positive": MetaField.POSITIVE_PROMPT, "negative": MetaField.NEGATIVE_PROMPT}.items():
                ref = prompt_node["inputs"].get(label)
                if isinstance(ref, list):
                    node = prompt.get(ref[0])
                    if node and isinstance(node.get("inputs"), dict):
                        text = node["inputs"].get("text")
                        if isinstance(text, str):
                            result_dict[meta].append((ref[0], text, 0))

    @classmethod
    def extract_model_info(cls, inputs, meta_field_name, prefix):
        model_info_dict = {}
        model_names = inputs.get(meta_field_name, [])
        model_hashes = inputs.get(f"{meta_field_name}_HASH", [])

        for index, (model_name, model_hash) in enumerate(zip(model_names, model_hashes)):
            field_prefix = f"{prefix}_{index}"
            model_info_dict[f"{field_prefix} name"] = os.path.splitext(os.path.basename(model_name[1]))[0]
            model_info_dict[f"{field_prefix} hash"] = model_hash[1]

        return model_info_dict

    @classmethod
    def gen_loras(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.LORA_MODEL_NAME, "Lora")

    @classmethod
    def gen_embeddings(cls, inputs):
        return cls.extract_model_info(inputs, MetaField.EMBEDDING_NAME, "Embedding")

    @classmethod
    def gen_parameters_str(cls, pnginfo_dict):
        def clean_value(value):
            if value is None:
                return ""
            value = str(value).strip()
            return value.replace("\n", " ")

        cleaned_dict = {k: clean_value(v) for k, v in pnginfo_dict.items()}

        result = [cleaned_dict.get("Positive prompt", "")]
        negative_prompt = cleaned_dict.get("Negative prompt")
        if negative_prompt:
            result.append(f"Negative prompt: {negative_prompt}")

        s_list = [
            f"{k}: {v}"
            for k, v in cleaned_dict.items() 
            if k not in {"Positive prompt", "Negative prompt"} and v not in {None, ""}
        ]

        result.append(", ".join(s_list))
        return "\n".join(result)

    @classmethod
    def get_hashes_for_civitai(cls, inputs_before_sampler_node, inputs_before_this_node):
        def extract_single(inputs, key):
            items = inputs.get(key, [])
            return items[0][1] if items and len(items[0]) > 1 else None

        def extract_named_hashes(names, hashes, prefix):
            result = {}
            for name, h in zip(names, hashes):
                base_name = os.path.splitext(os.path.basename(name[1]))[0]
                result[f"{prefix}:{base_name}"] = h[1]
            return result

        resource_hashes = {}

        model = extract_single(inputs_before_sampler_node, MetaField.MODEL_HASH)
        if model:
            resource_hashes["model"] = model

        vae = extract_single(inputs_before_this_node, MetaField.VAE_HASH)
        if vae:
            resource_hashes["vae"] = vae
            
        upscaler_hash = extract_single(inputs_before_this_node, MetaField.UPSCALE_MODEL_HASH)
        if upscaler_hash:
            resource_hashes["upscaler"] = upscaler_hash

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_NAME, []),
            inputs_before_sampler_node.get(MetaField.LORA_MODEL_HASH, []),
            "lora"
        ))

        resource_hashes.update(extract_named_hashes(
            inputs_before_sampler_node.get(MetaField.EMBEDDING_NAME, []),
            inputs_before_sampler_node.get(MetaField.EMBEDDING_HASH, []),
            "embed"
        ))

        return resource_hashes

    @classmethod
    def get_sampler_for_civitai(cls, sampler_names, schedulers):
        """
        Get the pretty sampler name for Civitai in the form of `<Sampler Name> <Scheduler name>`.
            - `dpmpp_2m` and `karras` will return `DPM++ 2M Karras`
        
        If there is a matching sampler name but no matching scheduler name, return only the matching sampler name.
            - `dpmpp_2m` and `exponential` will return only `DPM++ 2M`

        if there is no matching sampler and scheduler name, return `<sampler_name>_<scheduler_name>`
            - `ipndm` and `normal` will return `ipndm`
            - `ipndm` and `karras` will return `ipndm_karras`

        Reference: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts
        """

        # Sampler map: https://github.com/civitai/civitai/blob/fe76d9a4406d0c7b6f91f7640c50f0a8fa1b9f35/src/server/common/constants.ts#L699
        sampler_dict = {
            'euler': 'Euler',
            'euler_ancestral': 'Euler a',
            'heun': 'Heun',
            'dpm_2': 'DPM2',
            'dpm_2_ancestral': 'DPM2 a',
            'lms': 'LMS',
            'dpm_fast': 'DPM fast',
            'dpm_adaptive': 'DPM adaptive',
            'dpmpp_2s_ancestral': 'DPM++ 2S a',
            
            'dpmpp_sde': 'DPM++ SDE',
            'dpmpp_sde_gpu': 'DPM++ SDE',
            'dpmpp_2m': 'DPM++ 2M',
            'dpmpp_2m_sde': 'DPM++ 2M SDE',
            'dpmpp_2m_sde_gpu': 'DPM++ 2M SDE',
            
            'ddim': 'DDIM',
            'plms': 'PLMS',
            'uni_pc': 'UniPC',
            'uni_pc_bh2': 'UniPC',
            'lcm': 'LCM'
        }

        sampler = None
        scheduler = None
        # Get the sampler and scheduler values
        if len(sampler_names) > 0:
            sampler = sampler_names[0][1]
        if len(schedulers) > 0:
            scheduler = schedulers[0][1]

        def get_scheduler_name(sampler_name, scheduler):
            if scheduler == "karras":
                return f"{sampler_name} Karras"
            elif scheduler == "exponential":
                return f"{sampler_name} Exponential"
            elif scheduler == "normal":
                return sampler_name
            else:
                return f"{sampler_name}_{scheduler}"

        if not sampler:
            return None
    
        if sampler in sampler_dict:
            return get_scheduler_name(sampler_dict[sampler], scheduler)

        # If no match in the dictionary, return the sampler name with scheduler appended
        return get_scheduler_name(sampler, scheduler)
