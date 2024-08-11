"""
WIP - not working yet
"""

import torch
import torch.nn as nn
from torch import Tensor

from lstm_crf.util import argmax, log_sum_exp

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class TransformerCrf(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        tag_to_ix: dict[str, int],
        # embedding_dim: int,
        hidden_dim: int,
    ):
        super(TransformerCrf, self).__init__()
        # self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=1)

        # Maps the output of the transformer into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # self.init_hidden()

    # def init_hidden(self) -> None:
    #     self.hidden = (
    #         torch.randn(2, 1, self.hidden_dim // 2),
    #         torch.randn(2, 1, self.hidden_dim // 2),
    #     )

    def neg_log_likelihood(self, sentence: Tensor, tags: Tensor) -> Tensor:
        """
        Get the NLL of a particular sentence and tag/state sequence.

        Returns a scalar tensor.
        """
        feats = self._nn_forward(sentence)
        forward_score = self._crf_forward(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence: Tensor) -> tuple[Tensor, list[int]]:
        """Full forward pass on the network including both the transformer and CRF parts."""
        # dont confuse this with _forward_alg below.
        # get the emission scores from the nn
        nn_feats = self._nn_forward(sentence)

        # find the best path, given the features.
        score, tag_seq = self._viterbi_decode(nn_feats)
        return score, tag_seq

    def _transformer_forward(self, sentence: Tensor) -> Tensor:
        """Forward pass through the transformer part of the network."""
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        transformer_out = self.transformer(embeds, embeds)
        transformer_out = transformer_out.view(len(sentence), self.hidden_dim)
        return transformer_out

    def _nn_forward(self, sentence: Tensor) -> Tensor:
        """Forward pass through the LSTM part of the network."""
        lstm_out = self._transformer_forward(sentence)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _crf_forward(self, feats: Tensor) -> Tensor:
        """
        'Forward' algorithm on the CRF part.

        Input is the hidden layer of the LSTM as "features".

        Returns a scalar tensor.
        """
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.0)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats: Tensor, tags: Tensor) -> Tensor:
        """
        Given "features" (the output of the LSTM), compute the score for the tag sequence.
        """
        # returns a scalar
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags]
        )
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats: Tensor) -> tuple[Tensor, list[int]]:
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.0)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
