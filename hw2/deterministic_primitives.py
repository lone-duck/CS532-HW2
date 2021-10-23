import torch
import math
import operator as op

def pure_hashmap_update(d, k, v):
    if isinstance(k, torch.Tensor):
        k = k.item()
    d2 = d.copy()
    d2.update({k:v})
    return d2
    

# inspired by https://norvig.com/lispy.html
def eval_env():
	env = {}
	env.update({
		'+': op.add,
       	'-': op.sub,
        '*': op.mul,
        '/': op.truediv,
        '>': op.gt,
        '<': op.lt,
        '>=': op.ge,
        '<=': op.le, 
        '=': op.eq,
        'sqrt': torch.sqrt,
        'vector': lambda *x: torch.tensor(x),
        'hash-map': lambda *x : dict(zip([i.item() if isinstance(i, torch.Tensor) else i for i in x[::2]], x[1::2])),
        'get': lambda x, y: x[y.long()] if isinstance(x, torch.Tensor) else x[y.item() if isinstance(y, torch.Tensor) else y],
        'put': lambda x, y, z: torch.cat((x[:y.long()], torch.tensor([z]), x[y.long()+1:])) if isinstance(x, torch.Tensor) else pure_hashmap_update(x,y,z),
        'append' : lambda x, y: torch.cat((x, torch.tensor([y]))),
        'first' : lambda x: x[0],
        'last' : lambda x: x[-1],
        'remove': lambda x, y : torch.cat((x[:y.long()], x[y.long()+1:])) if isinstance(x, torch.Tensor) else {i:x[i] for i in x if i != y}
        })


	return env