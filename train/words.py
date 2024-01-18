import torch
from torch.nn.functional import one_hot, cross_entropy, softmax
from torch.optim import *
from torch.utils.data import DataLoader
import torchvision

from PIL import Image
import numpy as np

with open("data/words/dataset.pickle", "rb") as out_file:
    image_tensor, labels = torch.load(out_file)

symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

n_words = len(image_tensor)
X = torch.empty((n_words * 5, 3, 50, 50))
Y = torch.empty((n_words * 5, 36))
index = 0
for image_i in range(n_words):
    for character_i in range(5):
        character, left, right = labels[image_i][character_i]
        
        m = int((left + right) / 2)
        x0 = m - 25
        if x0 < 0:
            x0 = 0
        if x0 > 199:
            x0 = 199
        
        X[index] = image_tensor[image_i][:, x0:x0 + 50, :]
        y = symbols.index(character)
        y = torch.tensor(y)
        Y[index] = one_hot(y, 36)

        index += 1

# Image.fromarray(np.array(data_x[10].swapaxes(0, 2)).astype(np.uint8)).show()
# print(symbols[int(data_y[10].argmax())])
        
X = X.to("cuda")
Y = Y.to("cuda")

n_train = 9000
n_epochs = 60
batch_size = 64

X_train = X[:n_train]
Y_train = Y[:n_train]
X_test  = X[n_train:]
Y_test  = Y[n_train:]

loader = DataLoader(list(range(n_train)), batch_size=batch_size)
n_batches = len(loader)

model = torchvision.models.mobilenet_v3_small(num_classes=26 + 10)
model = model.to("cuda")
optimizer = Adam(model.parameters())

batch_losses = torch.empty((n_batches,))

print("Training...")
model.train()
try:
    for epoch_number in range(1, n_epochs + 1):
        print("epoch", epoch_number)
        for i, batch in enumerate(loader):
            batch_x = X_train[batch]
            batch_y = Y_train[batch]

            y_hat = model(batch_x)
            loss = cross_entropy(y_hat, batch_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses[i] = loss

            print(f"[{i + 1}/{n_batches}] - {loss:.6f}", end="           \r")

        model.eval()
        with torch.no_grad():
            logits = model(X_test[:1000])
            p = softmax(logits, dim=-1)
            predictions = p.argmax(dim=1)
            actual_labels = Y_test[:1000].argmax(dim=1)
            test_loss = float(cross_entropy(logits, Y_test[:1000]))
            test_accuracy = (predictions == actual_labels).to(float).mean()
            test_accuracy = float(test_accuracy)

            print("  train loss:", float(batch_losses.mean()))
            print("  val loss:", test_loss)
            print("  val acc:", test_accuracy)
        model.train()

        checkpoint = model.state_dict()
finally:
    print()

with open(f"weights/words.state", "wb") as out_file:
    torch.save(checkpoint, out_file)
