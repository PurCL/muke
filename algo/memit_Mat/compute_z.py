import os
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.matryoshka import analyze_affinity, get_token_coeffs

from .memit_Mat_hparams import MEMITMatHyperParams


def compute_affinity_accumulate(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: MEMITMatHyperParams,
):

    # prepare path
    save_dir = f"output/cache/{hparams.alg_name}-{hparams.model_name}-{hparams.ds_name}-affinity-accum-cache-win{hparams.window_size}"
    os.makedirs(save_dir, exist_ok=True)
    save_pt_path = os.path.join(save_dir, f"{data['id']}.pt")
    if os.path.exists(save_pt_path):
        all_affinities = torch.load(save_pt_path)
        return analyze_affinity(all_affinities)
    
    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # Tokenize target into list of int token IDs
    target_ids = tok(data["answer"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    
    input_tok = tok(
        [data["question"]],  
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    # ============================ pre edit ==================================
    # derive a list of start-end indices for each window
    all_affinities = []
    
    
    ## chunk window
    start_end_indices = []
    start = 0
    while start < len(target_ids):
        end = start + hparams.window_size
        if end > len(target_ids):
            end = len(target_ids)
        start_end_indices.append((start, end))
        start += hparams.window_size - hparams.overlap
    
    # iterate over each window
    assert hparams.overlap == 0
    input_ids = torch.cat([input_tok['input_ids'] , target_ids.unsqueeze(0)], dim=1)
    input_ids = input_ids[:, :-1]
    ex_len = input_ids.size(1)  # seq_len
    
    
    # Optimize and collect dynamic delta for query window 
    # then compute gradients for other windows
    for query_window_idx in range(len(start_end_indices) - 1):  # the last window requires no computation
        
        current_target_ids = target_ids[start_end_indices[query_window_idx][0]:]
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            1, input_ids.size(1)
        )
        rewriting_targets[0, ex_len - len(current_target_ids) :] = current_target_ids
        lookup_idxs = \
            [ex_len + start_end_indices[query_window_idx][0] - len(target_ids)]
        loss_layer = max(hparams.v_loss_layer, layer)
        
        lm_config = model.config
        
        if hasattr(lm_config, 'n_embd'):
            delta = torch.zeros((lm_config.n_embd,), requires_grad=True, device="cuda")
        elif hasattr(lm_config, 'hidden_size'):
            delta = torch.zeros((lm_config.hidden_size,), requires_grad=True, device="cuda")
        else:
            raise NotImplementedError
        
        target_init = None
        def edit_output_fn(cur_out, cur_layer):
            "add delta to the located layer output"
            nonlocal target_init
            if cur_layer == hparams.layer_module_tmp.format(layer):
                if target_init is None:
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                for i, idx in enumerate(lookup_idxs):   # new delta, with grad
                    if len(lookup_idxs)!=len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta
            return cur_out
        
        # Optimizer
        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)
        
        # Execute optimization
        optim_steps_for_dynamics = 3
        query_delta_grads = []
        query_delta_states = []
        for it in range(optim_steps_for_dynamics):
            opt.zero_grad()

            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(input_ids).logits

            # compute window loss
            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            if output.shape[1] != rewriting_targets.shape[1]:
                output=torch.transpose(output, 0, 1)
            full_repr = output
            
            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            target_log_probs = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            token_coeffs, n_tokens = get_token_coeffs(
                start_end_indices,
                rewriting_targets[0],
                input_tok['input_ids'].size(1) - 1, # NOTE: last token is some padding token
                query_window_idx,
                "AnyEdit",
            )
            # print('coeffs:', token_coeffs)
            # print('n_tokens:', n_tokens)
            nll_loss_each = -(target_log_probs * token_coeffs.to(target_log_probs.device)).sum(dim=1) / n_tokens
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            loss = nll_loss + weight_decay.to(nll_loss.device)
            nll_loss = nll_loss.item()
            weight_decay = weight_decay.item()
            avg_prob = torch.exp(-nll_loss_each).mean().item()
            print(
                "[Affinity Query] "
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss, 3)} + {np.round(weight_decay, 3)} "
                f"avg prob {avg_prob}"
            )
            
            if it == optim_steps_for_dynamics:
                break

            # Backpropagate
            loss.backward()
            query_delta_grads.append(delta.grad.clone().cpu())
            query_delta_states.append(delta.clone().detach())
            opt.step()

        
        if len(query_delta_grads) < optim_steps_for_dynamics:
            assert 0    # debug
            query_delta_grads += [torch.zeros_like(query_delta_grads[0]) \
                for _ in range(optim_steps_for_dynamics - len(query_delta_grads))]
            query_delta_states += [query_delta_states[-1].clone() \
                for _ in range(optim_steps_for_dynamics - len(query_delta_states))]
        
        # compute window grads based on delta state trace
        key_delta_grads = []
        # compute
        query_start = start_end_indices[query_window_idx][0]
        for window_idx, (start, end) in enumerate(start_end_indices):
            if window_idx <= query_window_idx:
                continue
            current_target_ids = target_ids[query_start:]
            remaining_len = len(target_ids[end:])
            rewriting_targets = torch.tensor(-100, device="cuda").repeat(
                1, input_ids.size(1)
            )
            rewriting_targets[0, ex_len - len(current_target_ids) :] = current_target_ids
            if remaining_len > 0:   # NOTE: -0 = 0
                rewriting_targets[0, - remaining_len:] = -100
            __key_delta_grads = []
            for delta in query_delta_states:
                
                delta.requires_grad = True
                
                with nethook.TraceDict(
                    module=model,
                    layers=[
                        hparams.layer_module_tmp.format(loss_layer),
                        hparams.layer_module_tmp.format(layer),
                    ],
                    retain_input=False,
                    retain_output=True,
                    edit_output=edit_output_fn,
                ) as tr:
                    logits = model(input_ids).logits
                
                # compute window loss
                output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
                if output.shape[1] != rewriting_targets.shape[1]:
                    output=torch.transpose(output, 0, 1)
                full_repr = output
                
                log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
                target_log_probs = torch.gather(
                    log_probs,
                    2,
                    torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
                ).squeeze(2)
                token_coeffs, n_tokens = get_token_coeffs(
                    start_end_indices,
                    rewriting_targets[0],
                    input_tok['input_ids'].size(1) - 1, # NOTE: last token is some padding token
                    window_idx,
                    "default",
                )
                # print('coeffs:', token_coeffs)
                # print('n_tokens:', n_tokens)
                # breakpoint()
                nll_loss_each = -(target_log_probs * token_coeffs.to(target_log_probs.device)).sum(dim=1) / n_tokens
                nll_loss = nll_loss_each.mean()
                
                weight_decay = hparams.v_weight_decay * (
                    torch.norm(delta) / torch.norm(target_init) ** 2
                )
                loss = nll_loss + weight_decay.to(nll_loss.device)
                if torch.isnan(loss):
                    breakpoint()
                nll_loss = nll_loss.item()
                weight_decay = weight_decay.item()
                avg_prob = torch.exp(-nll_loss_each).mean().item()
                print(
                    "[Affinity Key] "
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss, 3)} + {np.round(weight_decay, 3)} "
                    # f"avg prob of [{cur_sen}] "
                    f"avg prob {avg_prob}"
                )
                
                # Backpropagate
                loss.backward()
                __key_delta_grads.append(delta.grad.clone().cpu())
                
                # zero grad of delta
                delta.grad.zero_()
            
            if __key_delta_grads:
                # __key_delta_grads = torch.stack(__key_delta_grads)
                # mean_accumulated_grads = torch.cumsum(
                #     __key_delta_grads, dim=0
                # ) / torch.arange(1, __key_delta_grads.size(0) + 1).unsqueeze(1)
                # key_delta_grads.append(mean_accumulated_grads)
                key_delta_grads.append(torch.stack(__key_delta_grads))
            
        # compute figure affinity
        q = torch.stack(query_delta_grads)
        q = (q / q.norm(dim=-1, keepdim=True)).unsqueeze(1)
        k = torch.stack(key_delta_grads)
        k_T = (k / k.norm(dim=-1, keepdim=True)).permute((1, 2, 0))
        affinity = torch.matmul(q, k_T).squeeze(1)
        all_affinities.append(affinity)
        
    torch.save(all_affinities, save_pt_path)

    return analyze_affinity(all_affinities)


def compute_z_matryoshka(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: MEMITMatHyperParams,
    affinities: list,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    #print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(data["answer"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    
    input_tok = tok(
        [data["question"]],  
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    all_delta = []
    all_target = []
    all_idxs = []
    
    
    # derive a list of start-end indices for each window
    start_end_indices = []
    start = 0
    while start < len(target_ids):
        end = start + hparams.window_size
        if end > len(target_ids):
            end = len(target_ids)
        start_end_indices.append((start, end))
        start += hparams.window_size - hparams.overlap

    # iterate over each window
    assert hparams.overlap == 0
    input_ids = torch.cat([input_tok['input_ids'] , target_ids.unsqueeze(0)], dim=1)
    input_ids = input_ids[:, :-1]
    
   
    # ============================  edit ==================================

    # log-specific
    log_r_target = torch.tensor(-100, device="cuda").repeat(
        1, input_ids.size(1)
    )
    log_r_target[0, -len(target_ids) :] = target_ids
    
    for window_idx, (start, end) in enumerate(start_end_indices):
        current_target_ids = target_ids[start:]
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            1, input_ids.size(1)
        )
        ex_len = input_ids.size(1)  # seq_len
        rewriting_targets[0, ex_len - len(current_target_ids) :] = current_target_ids
        lookup_idxs = [ex_len - len(current_target_ids)]
        loss_layer = max(hparams.v_loss_layer, layer)

        lm_config = model.config

        if hasattr(lm_config, 'n_embd'):
            delta = torch.zeros((lm_config.n_embd,), requires_grad=True, device="cuda")
        elif hasattr(lm_config, 'hidden_size'):
            delta = torch.zeros((lm_config.hidden_size,), requires_grad=True, device="cuda")
        else:
            raise NotImplementedError
        
        target_init = None
        def edit_output_fn(cur_out, cur_layer):
            
            nonlocal target_init
            
            if cur_layer == hparams.layer_module_tmp.format(layer):
                if target_init is None:
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                for idxs_pre, delta_pre in all_delta:   # early deltas, no grad
                    for i, idx in enumerate(idxs_pre):
                        if len(idxs_pre)!=len(cur_out[0]):
                            cur_out[0][idx, i, :] += delta_pre
                        else:
                            cur_out[0][i, idx, :] += delta_pre
                for i, idx in enumerate(lookup_idxs):   # new delta, with grad
                    if len(lookup_idxs)!=len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta

            return cur_out
        
        # Optimizer
        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)
        
        # Execute optimization
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(input_ids).logits

            # compute matryoshka loss
            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            if output.shape[1] != rewriting_targets.shape[1]:
                output=torch.transpose(output, 0, 1)
            full_repr = output
            
            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            target_log_probs = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            token_coeffs, n_tokens = get_token_coeffs(
                start_end_indices,
                rewriting_targets[0],
                input_tok['input_ids'].size(1) - 1, # NOTE: last token is some padding token
                window_idx,
                hparams.coeff_strategy,
                affinities[window_idx],
            )
            # print('coeffs:', token_coeffs)
            # print('n_tokens:', n_tokens)
            nll_loss_each = -(target_log_probs * token_coeffs.to(target_log_probs.device)).sum(dim=1) / n_tokens
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            loss = nll_loss + weight_decay.to(nll_loss.device)
            nll_loss = nll_loss.item()
            weight_decay = weight_decay.item()
            avg_prob = torch.exp(-nll_loss_each).mean().item()
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss, 3)} + {np.round(weight_decay, 3)} "
                # f"avg prob of [{cur_sen}] "
                f"avg prob {avg_prob}"
            )
            

            if loss < 1e-2:
               break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()
            opt.step()
            

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

            
        # cur_sen = ""
        target = target_init + delta  
        all_delta.append((lookup_idxs, delta.clone()))
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])
        print(
            f"Iteration {len(all_delta)}: Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
        )

    return all_idxs, all_target