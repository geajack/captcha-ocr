import torch
from torch.nn.functional import one_hot, cross_entropy, softmax
from torch.optim import *
from torch.utils.data import DataLoader
import numpy
import torchvision
from PIL import Image

symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

model = torchvision.models.mobilenet_v3_small(num_classes=26 + 10)
model = model.to("cuda")

dataset_name = "letters10"

try:
    with open(f"data/{dataset_name}/dataset.pickle", "rb") as file:
        X, Y = torch.load(file)
    print("Using cached dataset")
except:
    xs = []
    ys = []
    with open(f"data/{dataset_name}/labels.csv") as csv_file:
        for line in csv_file:
            line = line.strip()
            if len(line) == 0:
                break

            filename, label = line.split(",")
            y = symbols.index(label)
            image = Image.open(f"data/{dataset_name}/" + filename)

            x = numpy.array(image, dtype=numpy.float32)
            x = x.swapaxes(0, 2)
            x = x[None]
            x = torch.tensor(x)

            xs.append(x)
            ys.append(y)


    X = torch.concatenate(xs)
    Y = torch.tensor(ys)
    Y = one_hot(Y, 36).to(torch.float)

    with open(f"data/{dataset_name}/dataset.pickle", "wb") as pickle_file:
        torch.save((X, Y), pickle_file)

X = X.to("cuda")
Y = Y.to("cuda")

n_train = 9000
n_epochs = 60
batch_size = 64

X_train = X[:n_train]
Y_train = Y[:n_train]
X_test  = X[n_train:]
Y_test  = Y[n_train:]

loader = DataLoader(range(n_train), batch_size=batch_size, shuffle=True)
n_batches = len(loader)

optimizer = Adam(model.parameters())

batch_losses = torch.empty((n_batches,))

print("Training...")
model.train()
try:
    for epoch_number in range(1, n_epochs + 1):
        batches = list(loader)

        print("epoch", epoch_number)
        for i, batch in enumerate(batches):
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
finally:
    print()

with open(f"weights/{dataset_name}.state", "wb") as out_file:
    torch.save(model.state_dict(), out_file)
