import json
import os

from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField

from nodes import NODE_CLASS_MAPPINGS
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

    @staticmethod
    def sanitize_name(name):
        return os.path.splitext(os.path.basename(name))[0].replace(' ', '_').replace(':', '_')

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

            clean_name = cls.sanitize_name(name[1])
            
            # LoRA strings for prompt and "Hashes" list
            lora_strings.append(f"<lora:{clean_name}:{weight[1]}>")
            lora_hashes_list.append(f"{clean_name}: {hash_val[1]}")

        lora_hashes_string = ", ".join(lora_hashes_list)
        return lora_strings, lora_hashes_string

    @classmethod
    def gen_pnginfo_dict(cls, inputs_before_sampler_node, inputs_before_this_node, save_civitai_sampler=True):
        pnginfo_dict = {}

        # Prompts
        cls.update_fields(inputs_before_sampler_node, pnginfo_dict, [
            (MetaField.POSITIVE_PROMPT, "Positive prompt"),
            (MetaField.NEGATIVE_PROMPT, "Negative prompt"),
        ])
        cls.append_lora_models(inputs_before_sampler_node, pnginfo_dict)

        # Basic metadata
        cls.update_fields(inputs_before_sampler_node, pnginfo_dict, [
            (MetaField.STEPS, "Steps"),
            (MetaField.CFG, "CFG scale"),
            (MetaField.SEED, "Seed"),
            (MetaField.CLIP_SKIP, "Clip skip"),
            (MetaField.MODEL_NAME, "Model"),
            (MetaField.MODEL_HASH, "Model hash"),
        ])

        # Check if Denoise is less than 1.0 and add if so
        denoise_value = cls._val(inputs_before_sampler_node, MetaField.DENOISE)
        if denoise_value and 0 < float(denoise_value) < 1:
            pnginfo_dict["Denoising strength"] = float(denoise_value)

        # Check for 'Hires upscale' or 'Hires upscaler' and always add Denoise field
        hires_upscale = cls._val(inputs_before_this_node, MetaField.UPSCALE_BY)
        hires_upscaler = cls._val(inputs_before_this_node, MetaField.UPSCALE_MODEL_NAME)

        if hires_upscale or hires_upscaler:
            if denoise_value:
                pnginfo_dict["Denoising strength"] = denoise_value

        # Sampler & Scheduler
        if save_civitai_sampler:
            sampler = cls.get_sampler_for_civitai(
                inputs_before_sampler_node.get(MetaField.SAMPLER_NAME, []),
                inputs_before_sampler_node.get(MetaField.SCHEDULER, [])
            )
        else:
            sampler = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME, [[None, ""]])[0][1]
            scheduler = inputs_before_sampler_node.get(MetaField.SCHEDULER, [[None, ""]])[0][1]
            if scheduler and scheduler != "normal":
                sampler += f"_{scheduler}"

        if sampler:
            pnginfo_dict["Sampler"] = sampler

        # Image size
        w = cls._val(inputs_before_sampler_node, MetaField.IMAGE_WIDTH)
        h = cls._val(inputs_before_sampler_node, MetaField.IMAGE_HEIGHT)
        if w and h:
            pnginfo_dict["Size"] = f"{w}x{h}"

        # VAE info
        cls._update(inputs_before_this_node, pnginfo_dict, MetaField.VAE_NAME, "VAE")
        cls._update(inputs_before_this_node, pnginfo_dict, MetaField.VAE_HASH, "VAE hash")
        
        # Add Hi-Res, based on https://github.com/civitai/civitai/blob/0c6a61b2d3ee341e77a357d4c08cf220e22b1190/src/server/common/model-helpers.ts#L33
        cls._update(inputs_before_this_node, pnginfo_dict, MetaField.UPSCALE_BY, "Hires upscale")
        cls._update(inputs_before_this_node, pnginfo_dict, MetaField.UPSCALE_MODEL_NAME, "Hires upscaler")
        
        # Add Lora hashes, based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/extensions-builtin/Lora/scripts/lora_script.py#L78
        _, lora_hashes = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)
        if lora_hashes:
            pnginfo_dict["Lora hashes"] = f'"{lora_hashes}"'

        # LoRA & embeddings detailed info
        pnginfo_dict.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo_dict.update(cls.gen_embeddings(inputs_before_sampler_node))

        # Civitai hash data
        civitai_hashes = cls.get_hashes_for_civitai(inputs_before_sampler_node, inputs_before_this_node)

        if civitai_hashes:
            pnginfo_dict["Hashes"] = json.dumps(civitai_hashes)

        return pnginfo_dict

    @classmethod
    def update_fields(cls, inputs, target_dict, metafield_pairs):
        for key, label in metafield_pairs:
            value = cls._val(inputs, key)
            if value:
                target_dict[label] = value

    @classmethod
    def _update(cls, inputs, target, meta_key, label):
        val = cls._val(inputs, meta_key)
        if val:
            target[label] = val

    @staticmethod
    def _val(inputs, key):
        return inputs.get(key, [[None, None]])[0][1]

    @classmethod
    def append_lora_models(cls, inputs, pnginfo_dict):
        prompt = pnginfo_dict.get("Positive prompt", "")
        
        # Append Lora models to the positive prompt, which is required for the Civitai website to parse and apply Lora weights.
        # Format: <lora:Lora_Model_Name:weight_value>. Example: <lora:Lora_Name_00:0.6> <lora:Lora_Name_01:0.8>
        lora_strings, _ = cls.get_lora_strings_and_hashes(inputs)
        if lora_strings:
            pnginfo_dict["Positive prompt"] = (prompt + " " + " ".join(lora_strings)).strip()


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
        def clean(value):
            return str(value).strip().replace("\n", " ") if value else ""

        cleaned = {k: clean(v) for k, v in pnginfo_dict.items()}
        parts = [cleaned.get("Positive prompt", "")]

        if neg := cleaned.get("Negative prompt"):
            parts.append(f"Negative prompt: {neg}")

        extras = [
            f"{k}: {v}" for k, v in cleaned.items()
            if k not in {"Positive prompt", "Negative prompt"} and v
        ]
        if extras:
            parts.append(", ".join(extras))

        return "\n".join(parts)

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
