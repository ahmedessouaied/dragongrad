import math
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    #Basic Operators

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+' )
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): return self * other

    def __neg__(self): return self * (-1)

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def __rtruediv__(self, other): return other * self**-1

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f'**{power}')
        def _backward():
            self.grad += (power * (self.data)**(power - 1)) * out.grad
        out._backward = _backward
        return out
    
    #Nonlinear Functions

    def sin(self):
        out = Value(math.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out
    
    def cos(self):
        out = Value(math.cos(self.data), (self,), 'cos')
        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    #Reverse-mode autodiff

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    #Visualization 

    def draw(self):
        """Alternative drawing method with even clearer layout"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Build the graph structure
        nodes = []
        edges = []
        
        def collect_nodes(v, level=0):
            node_id = id(v)
            if not any(n['id'] == node_id for n in nodes):
                nodes.append({
                    'id': node_id,
                    'value': v,
                    'level': level,
                    'x': 0,  # Will be set later
                    'y': level
                })
                
                for i, child in enumerate(v._prev):
                    edges.append((id(child), node_id))
                    collect_nodes(child, level - 1)
        
        collect_nodes(self)
        
        # Position nodes by level
        levels = {}
        for node in nodes:
            level = node['level']
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Set x positions
        for level, level_nodes in levels.items():
            for i, node in enumerate(level_nodes):
                node['x'] = i - (len(level_nodes) - 1) / 2
        
        # Draw nodes
        for node in nodes:
            v = node['value']
            x, y = node['x'] * 2, node['y'] * 1.5
            
            # Choose color based on operation
            if v._op:
                color = 'lightcoral'
                size = 0.8
            else:
                color = 'lightblue'
                size = 0.6
            
            # Draw node circle
            circle = plt.Circle((x, y), size, color=color, alpha=0.7, zorder=2)
            ax.add_patch(circle)
            
            # Add text
            if v._op:
                text = f"{v._op}\nval: {v.data:.2f}\n∇: {v.grad:.2f}"
            else:
                text = f"val: {v.data:.2f}\n∇: {v.grad:.2f}"
            
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=10, fontweight='bold', zorder=3)
        
        # Draw edges
        node_pos = {node['id']: (node['x'] * 2, node['y'] * 1.5) for node in nodes}
        for start_id, end_id in edges:
            start_pos = node_pos[start_id]
            end_pos = node_pos[end_id]
            
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                       zorder=1)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(min(node['y'] for node in nodes) - 1, 
                   max(node['y'] for node in nodes) + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Computation Graph (val = value, ∇ = gradient)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

x = Value(2.0)
y = Value(4.0)
z = x * y + x**2
z.backward()

print("Values after backward pass:")
print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")

print("\nDrawing computation graph...")
z.draw()