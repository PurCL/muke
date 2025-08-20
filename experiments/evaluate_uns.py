import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from time import time
from typing import Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from algo import *
from dataset_classes import DS_DICT, TEMPLATE_DICT
from util import nethook
from util.globals import DATA_DIR, HPARAMS_DIR
from util.hparams import HyperParams


def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_run_name(
    alg_name: str,
    model_name: str,
    ds_name: str,
    dataset_size_limit: int,
    sequential: bool,
    hparams: HyperParams,
    num_edits: int
):
    run_name = f"{alg_name}-{model_name}-{ds_name}-{dataset_size_limit}"

    if hasattr(hparams, 'v_num_grad_steps'):
        run_name += f"-step={hparams.v_num_grad_steps}"

    run_name += f"-NE={num_edits}"

    if sequential:
        run_name += "-sequential"

    if getattr(hparams, 'window_size', None):
        run_name += f"-win={hparams.window_size}"

    if getattr(hparams, 'arg_note', None):
        run_name += f"-{hparams.arg_note}"

    run_name += f"-{datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d-%H%M%S')}"
        
    return run_name


def tokenize_with_model_settings(tokenizer, input_text, model_name, **kwargs):
    """
    Helper function to ensure proper tokenization settings for different models
    """
    # Default to pytorch tensors if not specified
    if 'return_tensors' not in kwargs:
        kwargs['return_tensors'] = 'pt'
        
    return tokenizer(input_text, **kwargs)



def strip_eos(model_name: str, tokenizer: AutoTokenizer, answer: str):
   
    if model_name in ['Llama3-8B-Instruct', 'Qwen2.5-7B-Instruct']:
        return answer.removesuffix(tokenizer.eos_token)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int = 1,
    sequential: bool = False,
):
    set_seed()
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    params_path = (HPARAMS_DIR / alg_name / hparams_fname)
    
    if hparams_fname.endswith('.yaml'):
        hparams = params_class.from_hparams(params_path)
    else:
        hparams = params_class.from_json(params_path)

    run_name = get_run_name(
        alg_name,
        model_name,
        ds_name,
        dataset_size_limit,
        sequential,
        hparams,
        num_edits
    )

    setattr(hparams, "run_name", run_name)
    setattr(hparams, "ds_name", ds_name)
    setattr(hparams, "alg_name", alg_name)

    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

        tok = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        if 'llama' in model_name.lower():
            # Llama 3 does not have a pad token set.
            tok.pad_token_id = tok.eos_token_id
                
        # Ensure pad token is set
        if not hasattr(tok, 'pad_token') or tok.pad_token is None:
            tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR,
                  model_name=hparams.model_name,
                  size=dataset_size_limit,
                  apply_template=(alg_name not in WISE_DICT.keys())
    )
    
    with open(Path(DATA_DIR)/"alpaca_data.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
        
    template = TEMPLATE_DICT[hparams.model_name]
    ex_datas = [
        template.wo_answer(
            i['instruction']+i['input']
        ) + i['output']
        for i in ex_datas
    ]
    
    # WISE
    if alg_name in WISE_DICT.keys():
        with open(Path(DATA_DIR)/"ZsRE"/"zsre_mend_train.json", 'r', encoding='utf-8') as json_file:
            wise_loc_data = json.load(json_file)
            wise_loc_prompts = [d['loc'] + ' ' + d['loc_ans'] for d in wise_loc_data]
            
        if len(wise_loc_data) < len(ds):
            wise_loc_prompts = (wise_loc_prompts * math.ceil(len(ds) / len(wise_loc_data)))
        ex_datas = wise_loc_prompts[:len(ds)]
        random.shuffle(ex_datas)
        
        for loc_prompt, data in zip(ex_datas, ds):
            data['loc_prompt'] = loc_prompt
        

    # Use the single tokenizer for all operations
    tokenizer = tok

    # AlphaEdit
    if alg_name in ALPHAEDIT_DICT.keys():
        raise NotImplementedError("Please refer to AnyEdit's implementation.")
    
    batch_size = num_edits
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size else 0)
    
    edited_data = []
    for batch_index in tqdm(range(num_batches), desc="Processing batches"):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = ds[start_index:end_index] 

        
        # unke
        if alg_name in UNKE_DICT.keys():
            ex_args = dict(ex_data = random.sample(ex_datas, 20))
        else:
            ex_args = dict()
        
        # AlphaEdit
        if alg_name in ALPHAEDIT_DICT.keys():
            raise NotImplementedError("Please refer to AnyEdit's implementation.")
        else:
            nc_args = dict()
        
 
        start = time()
        if alg_name in ALG_DICT.keys():
            
            # model is original model, weights_copy is the original model weights
            weights_copy = apply_algo(model, tok, hparams, batch, **ex_args, **nc_args)
            # model becomes the edited model

        else:
            raise NotImplementedError(f"Algorithm {alg_name} not implemented")
                
        tqdm.write(f"Execution took {time() - start} seconds")

        start = time()
        if not sequential:
            for data in batch:
                if ds_name in ['unke','cf']:
                    question = tokenize_with_model_settings(tokenizer, [data['question'],data['para_question']], hparams.model_name, padding=True)
                else:
                    question = tokenize_with_model_settings(tokenizer, [data['question']], hparams.model_name, padding=True)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids=question['input_ids'].to('cuda'),
                    attention_mask=question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                if batch_index < 10 // batch_size + 1:
                    tqdm.write(f"question:{data['question']}")
                    tqdm.write(output[0])
                    if ds_name in ['unke','cf']:
                        tqdm.write(f"question:{data['para_question']}")
                        tqdm.write(output[1])
    
                data['original_prediction'] = output[0]
                if ds_name in ['unke','cf']:
                    data['para_prediction'] = output[1]
                
                data['answer'] = strip_eos(hparams.model_name, tokenizer, data['answer'])

            if ds_name in ['unke','cf','mquake']:
                for data in batch:
                    question = tokenize_with_model_settings(tokenizer, data['sub_question'], hparams.model_name, padding=True)
                    with torch.no_grad():
                        generated_ids = model.generate(
                        input_ids=question['input_ids'].to('cuda'),
                        attention_mask=question['attention_mask'].to('cuda'),
                        do_sample=True,
                        temperature=0.001,# Analysis exp
                        max_new_tokens=512
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                    ]

                    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    data['sub_pred'] = output
         
            edited_data.extend(batch)
            
            if alg_name in WISE_DICT.keys():
                # weights_copy is WISE.reset_layer()
                weights_copy()
            else:
                with torch.no_grad(): # model is reverted to original weights
                    for k, v in weights_copy.items():
                        nethook.get_parameter(model, k)[...] = v.to("cuda")


    if sequential:
        for data in ds:
            if ds_name in ['unke','cf']:
                question = tokenize_with_model_settings(tokenizer, [data['question'],data['para_question']], hparams.model_name, padding=True)
            else:
                question = tokenize_with_model_settings(tokenizer, [data['question']], hparams.model_name, padding=True)
            #print(question.input_ids) 
            with torch.no_grad():
                generated_ids = model.generate(
                input_ids=question['input_ids'].to('cuda'),
                attention_mask=question['attention_mask'].to('cuda'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
            ]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if batch_index < 10 // batch_size + 1:
                print(f"question:{data['question']}")
                print(output[0])
                if ds_name in ['unke','cf']:
                    print(f"question:{data['para_question']}")
                    print(output[1])
            data['original_prediction'] = output[0]
            if ds_name in ['unke','cf']:
                data['para_prediction'] = output[1]

            data['answer'] = strip_eos(hparams.model_name, tokenizer, data['answer'])

        if ds_name in ['unke','cf','mquake']:
            for data in ds:
                question = tokenize_with_model_settings(tokenizer, data['sub_question'], hparams.model_name, padding=True)
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids=question['input_ids'].to('cuda'),
                    attention_mask=question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,# Analysis exp
                    max_new_tokens=512
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]

                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                data['sub_pred'] = output
        
        edited_data.extend(ds)
    
    
    path = f'output/{run_name}_result.json'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as json_file: 
        json.dump(edited_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"saving to {path}")

    print("Evaluation took", time() - start)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=ALG_DICT.keys(),
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. ",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="Llama3-8B-Instruct",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama3-8B-Instruct.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=DS_DICT.keys(),
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )

    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )

    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )

    parser.add_argument(
        "--sequential",
        dest="sequential",
        action="store_true",
        help="sequential editing",
    )

    args = parser.parse_args()



    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        num_edits=args.num_edits,
        sequential=args.sequential,
    )
