# 🐉 DragonGrad

A tiny **reverse-mode automatic differentiation engine** built from scratch in Python.  
Inspired by [micrograd](https://github.com/karpathy/micrograd), but with a twist:  
💡 **custom visualizations**, 🔍 **clear gradients**, and 🧠 **neural network support**.

---

## ✨ Features

- ✅ Reverse-mode autodiff engine  
- ✅ Operator overloading (`+`, `*`, `**`, `/`, etc.)  
- ✅ Elementary functions: `tanh`, `sin`, `cos`, `exp`  
- ✅ Custom gradient visualizations with `matplotlib`  
- ✅ Neural network training example  
- ✅ Clean API — plug-and-play with scalar values  
- ✅ Minimal, understandable code (under 300 lines!)  

---

## 📦 Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/ahmedessouaied/dragongrad.git
cd dragongrad
pip install -e .
```

Install required dependencies:

```bash
pip install matplotlib networkx
```

---

## 🧪 Example Usage

```python
from dragongrad import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x**2

z.backward()

print(f"z = {z.data}, dz/dx = {x.grad}, dz/dy = {y.grad}")
z.draw()  # visualize the computation graph
```

---

## 🧠 Neural Network Example

Run this to see training in action:

```bash
python examples/example_nn.py
```

It trains a small neural net to learn XOR-like patterns using `dragongrad`.

---

## 📈 Visualization

Each computation builds a graph. After `.backward()` you can call `.draw()`:

```python
z.draw()
```

This uses `matplotlib` to render:
- Node values
- Gradients
- Operation labels

---

## 📁 Project Structure

```
dragongrad/
│
├── dragongrad/
│   ├── engine.py       ← Core autodiff engine
│   └── __init__.py     ← Exports Value
│
├── examples/
│   └── example_nn.py   ← Neural net demo
│
├── tests/
│   └── test_engine.py  ← Unit tests
│
├── setup.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 🔭 Roadmap

- [ ] Add ReLU, Sigmoid, Log  
- [ ] Add multi-layer perceptron class  
- [ ] Gradient checking  
- [ ] Export graphs to PNG/SVG  
- [ ] Publish to PyPI  

---

## 👤 Author

**Ahmed Essouaied**  
🎓 2nd year ICT Engineering student  
📫 [Your GitHub or LinkedIn profile here]

---

## 📄 License

MIT License – see [`LICENSE`](./LICENSE) for details.