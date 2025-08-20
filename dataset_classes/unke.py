import json
from pathlib import Path

from util.globals import *
from .templates import (
    TEMPLATE_DICT,
    EOS_DICT
)

class UnKEDataset:

    def __init__(self, data_dir: str, model_name: str, apply_template: bool = True, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        with open(data_dir/"UnKE"/"final_data_v3.json", 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        if apply_template:
            assert model_name in TEMPLATE_DICT.keys(), f"Model {model_name} not implemented"
            template = TEMPLATE_DICT[model_name]
            
            for i in raw:
                i['question'] = template.wo_answer(i['question'])
                i['para_question'] = template.wo_answer(i['para_question'])
                i['answer'] = i['answer'] + EOS_DICT[model_name]
                i['sub_question'] = template.wo_answer_list(i['sub_question'], False)

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