import bohrium as bh
import inspect

a = bh.arange(9)

b = bh.where(a > 3, a, -1)

print(inspect.getsource(bh.where))
