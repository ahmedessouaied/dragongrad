from dragongrad import Value
import random

# Define a simple 1-hidden neuron neural net
class TinyNN:
    def __init__(self):
        self.w1 = [Value(random.uniform(-1, 1)) for _ in range(2)]
        self.b1 = Value(0.0)
        self.w2 = [Value(random.uniform(-1, 1)) for _ in range(2)]
        self.b2 = Value(0.0)

    def forward(self, x1, x2):
        h1 = self.w1[0] * x1 + self.w1[1] * x2 + self.b1
        h1 = h1.tanh()
        out = self.w2[0] * h1 + self.w2[1] * self.b1 + self.b2
        return out

# XOR-ish data
data = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]

net = TinyNN()
lr = 0.1

for epoch in range(100):
    total_loss = 0
    for x1_val, x2_val, y_val in data:
        x1, x2, y = Value(x1_val), Value(x2_val), Value(y_val)
        y_pred = net.forward(x1, x2)
        loss = (y_pred - y) ** 2

        for p in net.w1 + [net.b1] + net.w2 + [net.b2]:
            p.grad = 0
        loss.backward()
        for p in net.w1 + [net.b1] + net.w2 + [net.b2]:
            p.data -= lr * p.grad

        total_loss += loss.data
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.4f}")

# Draw graph of final prediction
print("\nVisualizing computation graph of final prediction:")
y_pred.draw()
