import torch
import torch.distributions as dist

from daphne import daphne

from primitives import eval_env
from tests import is_tol, run_prob_test,load_truth

ENV = eval_env()

def deterministic_evaluate(e, l, sig=None):
    # variable reference OR procedure OR just a string
    if isinstance(e, str):        
        # global procedures take precedence over locally defined vars
        if e in ENV:
            return ENV[e], sig
        elif e in l:
            return l[e], sig
        # could allow for hashmaps with string keys; for debugging setting this to fail
        else:
            assert False, "Unknown symbol: {}".format(e)
    # constant number
    elif isinstance(e, (int, float)):   
        return torch.tensor(float(e)), sig
    # if statements
    elif e[0] == 'if':
        (_, test, conseq, alt) = e
        exp = (conseq if deterministic_evaluate(test, l)[0] else alt)
        return deterministic_evaluate(exp, l)
    # let statements
    elif e[0] == 'let':
        # get symbol
        symbol = e[1][0]
        # get value of e1
        value, _ = deterministic_evaluate(e[1][1], l)
        # evaluate e2 with value 
        return deterministic_evaluate(e[2], {**l, symbol: value})
    # sample statement
    if e[0] == 'sample':
        dist = deterministic_evaluate(e[1], l)[0]
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        return dist.sample(), sig
    # obsere statements
    # TODO: change this, maybe in this hw or for hw3
    if e[0] == 'observe':
        dist = deterministic_evaluate(e[1], l)[0] # get dist
        y = deterministic_evaluate(e[2], l)[0]    # get observed value
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        # TODO: do something with observed value
        return dist.sample(), sig
    # procedure call, either primitive or user-defined
    else:
        result = deterministic_evaluate(e[0], l)
        proc, sig = result
        # primitives are functions
        if callable(proc):
            args = [deterministic_evaluate(arg, l)[0] for arg in e[1:]]
            result, sig = proc(*args), sig
            return result, sig
        # user defined functions are not
        else:
            # as written in algorithm 6
            v_is, e0 = proc 
            assert(len(v_is) == len(e[1:]))
            c_is = [deterministic_evaluate(arg, l)[0] for arg in e[1:]]
            l_proc = dict(zip(v_is, c_is))
            return deterministic_evaluate(e0, {**l, **l_proc})

# inspired by https://www.geeksforgeeks.org/python-program-for-topological-sorting/
# TODO: update to python 3.9 and use graphlib instead
def topological_sort(A, V):
    visited = {v:False for v in V}
    stack = []
    
    for v in V:
        if visited[v] == False:
            topo_sort_util(v, A, V, visited, stack)
            
    return stack

def topo_sort_util(v, A, V, visited, stack):
    
    visited[v] = True
    
    if v in A:
        for adj_v in A[v]:
            if visited[adj_v] == False:
                topo_sort_util(adj_v, A, V, visited, stack)
            
    stack.insert(0, v)



def sample_from_joint(graph):
    """This function does ancestral sampling starting from the prior."""
    
    # get contents of graph
    fn_defs = graph[0]
    V = graph[1]['V']
    A = graph[1]['A']
    P = graph[1]['P']
    Y = graph[1]['Y']
    ret_vals = graph[2]
    
    # deal with fn_defs
    global ENV
    ENV = eval_env()
    for defn in fn_defs.items():
        f_name = defn[0]
        f_v_is = defn[1][1]
        f_expr = defn[1][2]
        ENV.update({f_name: (f_v_is, f_expr)})
    
    # get sorted V
    sorted_V = topological_sort(A, V)

    # compute each value in order
    l = {}
    for v in sorted_V:
        task, expr = P[v][0], P[v][1]
        if task == "sample*":
            dist, _ = deterministic_evaluate(expr, l)
            l.update({v: dist.sample()})
        # TODO: for now treat observes like samples; fix this later
        if task == "observe*":
            dist, _ = deterministic_evaluate(expr, l)
            l.update({v: dist.sample()})

    return deterministic_evaluate(ret_vals, l)[0]


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        
        
        ret = sample_from_joint(graph) 
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    
    run_probabilistic_tests()
    
    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))   

     

    