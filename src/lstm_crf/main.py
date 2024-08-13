import torch
import torch.optim as optim
from tqdm import tqdm

from lstm_crf.model import START_TAG, STOP_TAG, LstmCrf
from lstm_crf.transformer_crf import TransformerCrf
from lstm_crf.util import prepare_sequence

EMBEDDING_DIM = 5
HIDDEN_DIM = 4


TAG_TO_INDEX = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}


TRAINING_DATA = [
    (
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split(),
    ),
    ("georgia tech is a university in georgia".split(), "B I O O O O B".split()),
]


def main():

    word_to_index = build_word_index(TRAINING_DATA)

    model = LstmCrf(len(word_to_index), TAG_TO_INDEX, EMBEDDING_DIM, HIDDEN_DIM)
    # model = TransformerCrf(len(word_to_index), TAG_TO_INDEX, HIDDEN_DIM)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        sent1 = prepare_sequence(TRAINING_DATA[0][0], word_to_index)
        ref_tags1 = prepare_sequence(TRAINING_DATA[0][1], TAG_TO_INDEX)
        print(f"ref: {str(ref_tags1)}")
        print(f"before training: {model(sent1)}")

    for _ in tqdm(range(300)):
        for sentence, tags in TRAINING_DATA:
            # 1. Pytorch accumulates gradients, clear them out before each instance
            model.zero_grad()

            # 2. convert strings to IDs
            sentence_in = prepare_sequence(sentence, word_to_index)
            targets = prepare_sequence(tags, TAG_TO_INDEX)

            # 3. run the forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            print(loss)

            # 4. compute gradients, and update the parameters
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        print("FULL MODEL PREDICTION:")
        print(model(sent1))

        # print("INPUT:")
        # print(sent1)
        # # print("LSTM OUTPUT:")
        # # print(model._lstm_forward(sent1))
        # print("NN OUTPUT:")
        # print(model._nn_forward(sent1))

        # print("SOFTMAX:")
        # print(torch.softmax(model._nn_forward(sent1), dim=1))
        # print("ARGMAX:")
        # print(torch.argmax(model._nn_forward(sent1), dim=1))

        # print("TRANSITION WEIGHTS:")
        # print(model.transitions.data)
        # print(torch.softmax(model.transitions.data, dim=0))


def build_word_index(data):
    # give each word an ID
    word_to_index = {}
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
    return word_to_index


if __name__ == "__main__":
    main()
