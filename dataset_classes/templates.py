from collections import namedtuple

# def get_list_without_answer(que: list[str], cot: bool, model_short: str):
#     assert model_short in ['llama', 'qwen', 'phi']
#     func_name = f'get_{model_short}_without_answer'
#     if cot:
#         func_name = func_name + '_cot'
#     func = globals()[func_name]
#     return [func(line) for line in que]




########
# llama 3
########
def get_llama_with_answer(que, ans):
  return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ans}<|eot_id|>"""

def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_llama_without_answer_cot(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease provide a multi-hop explanation for the next question: {que}<|eot_id|>"""

def get_list_llama_without_answer(que, cot):
    if cot == False:
        L = [get_llama_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L

########
# qwen 2.5
########

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_qwen_without_answer_cot(que):
    return f"""<|im_start|>user\n Please provide a multi-hop explanation for the next question: {que}<|im_end|>\n<|im_start|>assistant\n"""

def get_list_qwen_without_answer(que, cot):
    if cot == False:
        L = [get_qwen_without_answer(line) for line in que]
    else:
        L = [get_qwen_without_answer_cot(line) for line in que]
    return L

########
# phi 4
########
# '<|im_start|>user<|im_sep|>somequestion<|im_end|><|im_start|>assistant<|im_sep|>'
def get_phi_without_answer(que):
    return f"""<|im_start|>user<|im_sep|>{que}<|im_end|><|im_start|>assistant<|im_sep|>"""

# '<|im_start|>user<|im_sep|>Please provide a multi-hop explanation for the next question: SOMEQUESTION<|im_end|><|im_start|>assistant<|im_sep|>'
def get_phi_without_answer_cot(que):
    return f"""<|im_start|>user<|im_sep|>Please provide a multi-hop explanation for the next question: {que}<|im_end|><|im_start|>assistant<|im_sep|>"""

def get_list_phi_without_answer(que, cot):
    if cot == False:
        L = [get_phi_without_answer(line) for line in que]
    else:
        L = [get_phi_without_answer_cot(line) for line in que]
    return L

########
# gemma 3
########
# '<bos><start_of_turn>user\nQUESTION<end_of_turn>\n<start_of_turn>model\n'
def get_gemma_without_answer(que):
    return f'<start_of_turn>user\n{que}<end_of_turn>\n<start_of_turn>model\n'

def get_gemma_without_answer_cot(que):
    return f'<start_of_turn>user\nPlease provide a multi-hop explanation for the next question: {que}<end_of_turn>\n<start_of_turn>model\n'

def get_list_gemma_without_answer(que, cot):
    if cot == False:
        L = [get_gemma_without_answer(line) for line in que]
    else:
        L = [get_gemma_without_answer_cot(line) for line in que]
    return L

########
# vicuna
########

def get_vicuna_without_answer(que):
    return f"""USER: {que} ASSISTANT:"""

Template = namedtuple("Template", ["wo_answer", "wo_answer_cot", "wo_answer_list"])

TEMPLATE_DICT = {
    "Llama3-8B-Instruct": Template(get_llama_without_answer, get_llama_without_answer_cot, get_list_llama_without_answer),
    "Qwen2.5-7B-Instruct": Template(get_qwen_without_answer, get_qwen_without_answer_cot, get_list_qwen_without_answer),
}

EOS_DICT = {
    "Llama3-8B-Instruct": "<|eot_id|>",
    "Qwen2.5-7B-Instruct": "<|im_end|>",
}