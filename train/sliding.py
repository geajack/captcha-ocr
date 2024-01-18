import torch
from torch.nn.functional import one_hot, cross_entropy, softmax
from torch.optim import *
from torch.utils.data import DataLoader
import torchvision

from PIL import Image
import numpy as np

from random import sample


SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?"


def show(data_x, data_y):
    top = data_y.topk(2)
    for i in range(2):
        print(SYMBOLS[top.indices[i]], float(top.values[i]))
    Image.fromarray(np.array(data_x.swapaxes(0, 2)).astype(np.uint8)).save("out.png")


class EpochData:

    def __init__(self, image_tensor, label_info, n_words_this_epoch=100, stride=2):
        n_words, _, image_w, image_h = image_tensor.shape

        window_width = image_h
        n_windows_per_word = (image_w - window_width) // stride
        self.n_windows = n_windows_per_word * n_words_this_epoch

        self.X = torch.empty((self.n_windows, 3, window_width, image_h), device="cuda")
        self.Y = torch.empty((self.n_windows, 37), device="cuda")

        word_indices = sample(list(range(n_words)), k=n_words_this_epoch)
        sample_i = 0
        for word_i in word_indices:
            window_xs = list(range(0, image_w - window_width, stride))
            assert len(window_xs) == n_windows_per_word, (len(window_xs), n_windows_per_word)

            for x in window_xs:
                self.X[sample_i] = image_tensor[word_i][:, x:x + window_width, :]

                window_m = int((x + x + window_width) / 2)
                prefix_m = int(label_info[word_i][0][1] / 2)
                suffix_m = int((label_info[word_i][-1][2] + image_w)/ 2)

                symbol = None

                if window_m <= prefix_m:
                    symbol = 36
                elif window_m >= suffix_m:
                    symbol = 36
                else:
                    left_m = prefix_m
                    right_m = suffix_m
                    left_symbol = 36
                    right_symbol = 36
                    for character_info in label_info[word_i]:
                        character, left, right = character_info
                        character_m = (left + right) // 2
                        if character_m == window_m:
                            symbol = SYMBOLS.index(character)
                            break                        
                        elif character_m < window_m:
                            left_m = character_m
                            left_symbol = SYMBOLS.index(character)
                        else:
                            right_m = character_m
                            right_symbol = SYMBOLS.index(character)
                            break

                y = torch.zeros((37,), dtype=torch.float)
                if symbol is None:
                    left_dx = window_m - left_m
                    right_dx = right_m - window_m
                    y[left_symbol] = right_dx
                    y[right_symbol] = left_dx
                    y /= left_dx + right_dx
                else:
                    y[symbol] = 1

                self.Y[sample_i] = y
                
                sample_i += 1

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_windows


with open("data/words/dataset.pickle", "rb") as out_file:
    image_tensor, labels = torch.load(out_file)

image_tensor = image_tensor.to("cuda", dtype=torch.float32)

n_epochs = 1000
batch_size = 512

model = torchvision.models.mobilenet_v3_small(num_classes=26 + 10 + 1)
model = model.to("cuda")
optimizer = Adam(model.parameters())

print("Training...")
model.train()
try:
    for epoch_number in range(1, n_epochs + 1):
        print("epoch", epoch_number)

        epoch = EpochData(image_tensor, labels, n_words_this_epoch=200, stride=4)
        loader = DataLoader(epoch, batch_size=batch_size)
        n_batches = len(loader)
        batch_losses = torch.empty((n_batches,))

        for i, batch in enumerate(loader):
            batch_x = batch[0]
            batch_y = batch[1]

            y_hat = model(batch_x)
            loss = cross_entropy(y_hat, batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses[i] = loss

            print(f"[{i + 1}/{n_batches}] - {loss:.6f}", end="           \r")

        print("  train loss:", float(batch_losses.mean()), "          ")

        # model.eval()
        # with torch.no_grad():
        #     logits = model(X_test[:1000])
        #     p = softmax(logits, dim=-1)
        #     predictions = p.argmax(dim=1)
        #     actual_labels = Y_test[:1000].argmax(dim=1)
        #     test_loss = float(cross_entropy(logits, Y_test[:1000]))
        #     test_accuracy = (predictions == actual_labels).to(float).mean()
        #     test_accuracy = float(test_accuracy)

        #     print("  val loss:", test_loss)
        #     print("  val acc:", test_accuracy)
        # model.train()

        checkpoint = model.state_dict()
finally:
    print()

with open(f"weights/sliding.state", "wb") as out_file:
    torch.save(checkpoint, out_file)