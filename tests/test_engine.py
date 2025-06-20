from dragongrad import Value

def test_add():
    x = Value(2.0)
    y = Value(3.0)
    z = x + y
    z.backward()
    assert z.data == 5.0
    assert x.grad == 1.0
    assert y.grad == 1.0
