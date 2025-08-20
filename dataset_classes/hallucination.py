import json
from pathlib import Path

from .templates import (
    TEMPLATE_DICT,
    EOS_DICT
)

class HallucinationDataset:

    def __init__(self, data_dir: str, model_name: str, apply_template: bool = True, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        with open(data_dir/"hallucination"/"hallucination-edit.json", 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        if apply_template:
            assert model_name in TEMPLATE_DICT.keys(), f"Model {model_name} not implemented"
            template = TEMPLATE_DICT[model_name]
            
            for i, d in enumerate(raw):
                d['id'] = i
                d['question'] = template.wo_answer(d['prompt'])
                d['answer'] = d['target_new']+EOS_DICT[model_name]

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