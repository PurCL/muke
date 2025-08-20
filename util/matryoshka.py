from typing import List, Tuple

import torch


def analyze_affinity(all_affinities: List):
    
    aggregated_affinites = []
    for affinities in all_affinities:

        affs = [1.0] + affinities.mean(dim=0).tolist()
                
        aggregated_affinites.append(affs)
    aggregated_affinites.append([1.0])
    return aggregated_affinites


def get_token_coeffs(
    start_end_indices: List[Tuple[int, int]],
    target_ids: List[int] | torch.Tensor,
    prefix_len: int,
    start_window_idx: int,
    strategy: str,
    affinity: torch.Tensor = None,
):
    """
    Computes the weights of the losses in the target sequence.
    """
    coeffs = torch.zeros_like(target_ids, dtype=torch.float16)

    if strategy == 'matryoshka':
        num_windows = len(start_end_indices) - start_window_idx
        one_start = start_end_indices[start_window_idx][0]
        for start, end in start_end_indices[start_window_idx:]:
            coeffs[prefix_len + one_start: prefix_len + end] += 1.0
        coeffs = coeffs / num_windows
        n_tokens = (coeffs != 0).sum().item()
    elif strategy == 'context-affinity':
        assert affinity is not None
        num_windows = len(start_end_indices) - start_window_idx
        one_start = start_end_indices[start_window_idx][0]
        for widx, (start, end) in enumerate(start_end_indices[start_window_idx:][::-1]):
            coeffs[prefix_len + one_start: prefix_len + end] += (2.0-affinity[widx])
        coeffs = coeffs / num_windows
        n_tokens = (coeffs != 0).sum().item()
    elif strategy == 'default':
        coeffs = (target_ids != -100).float()
        n_tokens = (coeffs != 0).sum().item()

    elif strategy == 'AnyEdit':
        start, end = start_end_indices[start_window_idx]
        coeffs[prefix_len + start: prefix_len + end] += 1.0
        n_tokens = (coeffs != 0).sum().item()
    else:
        raise NotImplementedError
    
    return coeffs, n_tokens