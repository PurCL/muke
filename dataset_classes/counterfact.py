import json
from pathlib import Path
from transformers import AutoTokenizer

from util.globals import *

from .templates import (
    get_llama_with_answer, get_llama_without_answer, get_llama_without_answer_cot,
    get_qwen_without_answer, get_qwen_without_answer_cot, 
    get_vicuna_without_answer, 
    get_list_llama_without_answer, get_list_qwen_without_answer,
    get_gemma_without_answer, get_gemma_without_answer_cot, get_list_gemma_without_answer,
    get_phi_without_answer, get_list_phi_without_answer
)


class CounterFactDataset:

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        with open(data_dir/"AKEW"/"CounterFact.json", 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        data = []
        for i, record in enumerate(raw):
            if model_name == 'Llama3-8B-Instruct':
                data.append(
                    {
                        "id": i,
                        "question": get_llama_without_answer(record["requested_rewrite"]["prompt_full"]),
                        "para_question": get_llama_without_answer(record["paraphrase_prompts"][0]),
                        "answer": record["requested_rewrite"]["fact_new_uns"]+'<|eot_id|>',
                        "sub_question": get_list_llama_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                        "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
                    }
                )
            elif model_name == 'Qwen2.5-7B-Instruct':
                data.append(
                    {
                        "id": i,
                        "question": get_qwen_without_answer(record["requested_rewrite"]["prompt_full"]),
                        "para_question": get_qwen_without_answer(record["paraphrase_prompts"][0]),
                        "answer": record["requested_rewrite"]["fact_new_uns"]+'<|im_end|>',
                        "sub_question": get_list_qwen_without_answer([q["prompt"].format(q["subject"]) for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5], False),
                        "sub_answer": [q["target"] for q in record["requested_rewrite"]["unsfact_triplets_GPT"]][:5]
                    }
                )
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")
        if size >= len(data):
            self._data = data
        else:
            import random
            random.shuffle(data)
            self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
