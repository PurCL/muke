import json
from pathlib import Path
from transformers import AutoTokenizer
from util.globals import *
from .templates import (
    get_llama_without_answer,
    get_qwen_without_answer,
    get_phi_without_answer,
    get_gemma_without_answer
)

class EditeveryDataset:

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        with open(data_dir/"editevery.json", 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        for i in raw:
            if model_name == 'Llama3-8B-Instruct':
                i['question'] = get_llama_without_answer(i['question'])
                i['answer'] = i['answer']+'<|eot_id|>'
            elif model_name == 'Qwen2.5-7B-Instruct':
                i['question'] = get_qwen_without_answer(i['question'])
                i['answer'] = i['answer']+'<|im_end|>'
            else:
                raise ValueError(f"Model {model_name} not supported")

        if size >= len(raw):
            self._data = raw
        else:
            import random
            random.shuffle(raw)
            self._data = raw[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
