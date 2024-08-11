import torch
import torch.optim as optim

from lstm_crf.main import EMBEDDING_DIM, HIDDEN_DIM, TAG_TO_INDEX, TRAINING_DATA, build_word_index
from lstm_crf.model import LstmCrf
from lstm_crf.util import prepare_sequence


def test_model():
 
    word_to_index = build_word_index(TRAINING_DATA)
    model = LstmCrf(len(word_to_index), TAG_TO_INDEX, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # check predictions before training
    with torch.no_grad():
        sent1 = prepare_sequence(TRAINING_DATA[0][0], word_to_index)
        ref_tags1 = prepare_sequence(TRAINING_DATA[0][1], TAG_TO_INDEX)
        print(f"ref: {str(ref_tags1)}")
        print(f"before training: {model(sent1)}")


    for _ in range(300):
        for sentence, tags in TRAINING_DATA:
            # 1. Pytorch accumulates gradients, clear them out before each instance
            model.zero_grad()

            # 2. convert strings to IDs
            sentence_in = prepare_sequence(sentence, word_to_index)
            targets = prepare_sequence(tags, TAG_TO_INDEX)

            # 3. run the forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # 4. compute gradients, and update the parameters
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        _, tags = model(sent1)
        assert tags == list(ref_tags1)