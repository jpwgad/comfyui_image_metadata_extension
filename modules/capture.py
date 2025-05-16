import json
import os
from collections import defaultdict
from . import hook
from .defs.captures import CAPTURE_FIELD_LIST
from .defs.meta import MetaField
from .defs.formatters import calc_lora_hash, calc_model_hash, extract_embedding_names, extract_embedding_hashes
from .utils.log import print_warning

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
        pnginfo = {}

        if not inputs_before_sampler_node:
            inputs_before_sampler_node = defaultdict(list)
            cls._collect_all_metadata(prompt, inputs_before_sampler_node)

        def extract(meta_key, label, source=inputs_before_sampler_node):
            """
            Scan the list behind `meta_key` and return the first payload that:
              1) is present (link has at least two elements),
              2) is not None,
              3) if it's a string, the string is not empty.
            Once found, it stores the value in `pnginfo[label]` and exits early.
            """
            # Retrieve the list associated with `meta_key`; default to an empty list
            val_list = source.get(meta_key, [])
            # Traverse the list in the original order (front to back)
            for link in val_list:
                # Guard against malformed link entries with length < 2
                if len(link) <= 1:
                    continue
                candidate = link[1]
                # Skip if None
                if candidate is None:
                    continue
                # If candidate is a string, skip empty ones
                if isinstance(candidate, str) and not candidate:
                    continue
                # First valid payload found – save and return immediately
                pnginfo[label] = candidate
                return candidate
            # No valid payload found
            return None

        # Prompts
        positive = extract(MetaField.POSITIVE_PROMPT, "Positive prompt") or ""
        if not positive.strip():
            print_warning("Positive prompt is empty!")

        negative = extract(MetaField.NEGATIVE_PROMPT, "Negative prompt") or ""
        lora_strings, lora_hashes = cls.get_lora_strings_and_hashes(inputs_before_sampler_node)
        # Append Lora models to the positive prompt, which is required for the Civitai website to parse and apply Lora weights.
        # Format: <lora:Lora_Model_Name:weight_value>. Example: <lora:Lora_Name_00:0.6> <lora:Lora_Name_01:0.8>
        if lora_strings:
            positive += " " + " ".join(lora_strings)

        pnginfo["Positive prompt"] = positive.strip()
        pnginfo["Negative prompt"] = negative.strip()

        # Sampling params
        if not extract(MetaField.STEPS, "Steps"):
            print_warning("Steps are empty, full metadata won't be added!")
            return {}  # No sense in pnginfo without the Steps parameter, ref https://github.com/civitai/civitai/blob/7c8f3f3044218cf3b3d86bd9f49d12fc196ea1f6/src/utils/metadata/automatic.metadata.ts#L102C42-L102C47

        samplers = inputs_before_sampler_node.get(MetaField.SAMPLER_NAME)
        schedulers = inputs_before_sampler_node.get(MetaField.SCHEDULER)

        if save_civitai_sampler:
            pnginfo["Sampler"] = cls.get_sampler_for_civitai(samplers, schedulers)
        elif samplers:
            sampler_name = samplers[0][1]
            if schedulers and schedulers[0][1] != "normal":
                sampler_name += f"_{schedulers[0][1]}"
            pnginfo["Sampler"] = sampler_name

        extract(MetaField.CFG, "CFG scale")
        extract(MetaField.SEED, "Seed")
        
        # Missing CLIP skip means it was set to 1 (the default)
        clip_skip = extract(MetaField.CLIP_SKIP, "Clip skip")
        if clip_skip is None:
            pnginfo["Clip skip"] = "1"

        # Image size
        image_width_data = inputs_before_sampler_node.get(MetaField.IMAGE_WIDTH, [[None]])
        image_height_data = inputs_before_sampler_node.get(MetaField.IMAGE_HEIGHT, [[None]])

        def extract_dimension(data):
            return data[0][1] if data and len(data[0]) > 1 and isinstance(data[0][1], int) else None

        width = extract_dimension(image_width_data)
        height = extract_dimension(image_height_data)
        if width and height:
            pnginfo["Size"] = f"{width}x{height}"

        # Model details
        extract(MetaField.MODEL_NAME, "Model")
        extract(MetaField.MODEL_HASH, "Model hash")
        extract(MetaField.VAE_NAME, "VAE", inputs_before_this_node)
        extract(MetaField.VAE_HASH, "VAE hash", inputs_before_this_node)

        # Denoising strength
        denoise = inputs_before_sampler_node.get(MetaField.DENOISE)
        dval = denoise[0][1] if denoise else None
        if dval and 0 < float(dval) < 1:
            pnginfo["Denoising strength"] = float(dval)

        # Include upscale info if present
        if inputs_before_this_node.get(MetaField.UPSCALE_BY) or inputs_before_this_node.get(MetaField.UPSCALE_MODEL_NAME):
            pnginfo["Denoising strength"] = float(dval or 1.0)

        # Hi-Res, based on https://github.com/civitai/civitai/blob/0c6a61b2d3ee341e77a357d4c08cf220e22b1190/src/server/common/model-helpers.ts#L33
        extract(MetaField.UPSCALE_BY, "Hires upscale", inputs_before_this_node)
        extract(MetaField.UPSCALE_MODEL_NAME, "Hires upscaler", inputs_before_this_node)

        # Lora hashes, based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/82a973c04367123ae98bd9abdf80d9eda9b910e2/extensions-builtin/Lora/scripts/lora_script.py#L78
        if lora_hashes:
            pnginfo["Lora hashes"] = f'"{lora_hashes}"'

        # Additional metadata
        pnginfo.update(cls.gen_loras(inputs_before_sampler_node))
        pnginfo.update(cls.gen_embeddings(inputs_before_sampler_node))

        hashes = cls.get_hashes_for_civitai(inputs_before_sampler_node, inputs_before_this_node)
        if hashes:
            pnginfo["Hashes"] = json.dumps(hashes)

        return pnginfo

    @classmethod
    def _collect_all_metadata(cls, prompt, result_dict):
        def _append_metadata(meta, node_id, value):
            if value is not None:
                result_dict[meta].append((node_id, value, 0))

        # Detect nodes with specific fields
        resolved = {
            "prompt": Trace.find_node_with_fields(prompt, {"positive", "negative"}),
            "denoise": Trace.find_node_with_fields(prompt, {"denoise"}),
            "sampler": Trace.find_node_with_fields(prompt, {"seed", "steps", "cfg", "sampler_name", "scheduler"}),
            "size": Trace.find_node_with_fields(prompt, {"width", "height"}),
            "model": Trace.find_node_with_fields(prompt, {"ckpt_name"}),
        }

        # LoRA metadata (multiple)
        for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"lora_name", "strength_model"}):
            if node is not None:
                inputs = node.get("inputs", {})
                name = inputs.get("lora_name")
                strength = inputs.get("strength_model")
                _append_metadata(MetaField.LORA_MODEL_NAME, node_id, name)
                _append_metadata(MetaField.LORA_MODEL_HASH, node_id, calc_lora_hash(name) if name else None)
                _append_metadata(MetaField.LORA_STRENGTH_MODEL, node_id, strength)

        # Model metadata
        model_node = resolved.get("model")
        if model_node and model_node[1] is not None:
            node_id, node = model_node
            inputs = node.get("inputs", {})
            name = inputs.get("ckpt_name")
            _append_metadata(MetaField.MODEL_NAME, node_id, name)
            _append_metadata(MetaField.MODEL_HASH, node_id, calc_model_hash(name) if name else None)

        # Denoise
        denoise_node = resolved.get("denoise")
        if denoise_node and denoise_node[1] is not None:
            node_id, node = denoise_node
            val = node.get("inputs", {}).get("denoise")
            _append_metadata(MetaField.DENOISE, node_id, val)

        # Sampler fields
        sampler_node = resolved.get("sampler")
        if sampler_node and sampler_node[1] is not None:
            node_id, node = sampler_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "sampler_name": MetaField.SAMPLER_NAME,
                "scheduler": MetaField.SCHEDULER,
                "seed": MetaField.SEED,
                "steps": MetaField.STEPS,
                "cfg": MetaField.CFG,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        # Image size fields
        size_node = resolved.get("size")
        if size_node and size_node[1] is not None:
            node_id, node = size_node
            inputs = node.get("inputs", {})
            for key, meta in {
                "width": MetaField.IMAGE_WIDTH,
                "height": MetaField.IMAGE_HEIGHT,
            }.items():
                _append_metadata(meta, node_id, inputs.get(key))

        # Prompt fields
        for node_id, node in Trace.find_all_nodes_with_fields(prompt, {"positive", "negative"}):
            if node is not None:
                inputs = node.get("inputs", {})
                pos_ref = inputs.get("positive", [None])[0]
                neg_ref = inputs.get("negative", [None])[0]

                def resolve_text(ref):
                    if isinstance(ref, list): ref = ref[0]
                    if not isinstance(ref, str): return None
                    node = prompt.get(ref)
                    return node.get("inputs", {}).get("text") if node else None

                pos_text = resolve_text(pos_ref)
                neg_text = resolve_text(neg_ref)
                
                # Append positive and negative prompts
                _append_metadata(MetaField.POSITIVE_PROMPT, pos_ref, pos_text)
                _append_metadata(MetaField.NEGATIVE_PROMPT, neg_ref, neg_text)

                # Add embedding metadata collection
                if pos_text:
                    embedding_names = extract_embedding_names(pos_text)
                    embedding_hashes = extract_embedding_hashes(pos_text)
                    for name, hash_ in zip(embedding_names, embedding_hashes):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, hash_)

                if neg_text:
                    embedding_names = extract_embedding_names(neg_text)
                    embedding_hashes = extract_embedding_hashes(neg_text)
                    for name, hash_ in zip(embedding_names, embedding_hashes):
                        _append_metadata(MetaField.EMBEDDING_NAME, node_id, name)
                        _append_metadata(MetaField.EMBEDDING_HASH, node_id, hash_)

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
        if not pnginfo_dict or not isinstance(pnginfo_dict, dict):
            return ""

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
        if sampler_names:
            if len(sampler_names) > 0:
                sampler = sampler_names[0][1]
                
        if schedulers:
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
