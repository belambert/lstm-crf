import torch
from torch import Tensor


def argmax(vec: Tensor) -> int:
    """return the argmax as a python int"""
    _, idx = torch.max(vec, 1)
    return int(idx.item())


def prepare_sequence(seq: list[str], to_ix: dict[str, int]) -> Tensor:
    """convert strings to a tensor of token IDs"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec: Tensor):
    """Compute log sum exp in a numerically stable way for the forward algorithm"""
    # get the max
    max_score = vec[0, argmax(vec)]
    # copy the max into a vector of size vec
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    assert vec.shape == max_score_broadcast.shape
    # do the log-sum-exp
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
