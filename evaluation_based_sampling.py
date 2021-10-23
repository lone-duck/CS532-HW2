from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import eval_env
import torch

ENV = None

def evaluate_program(ast, return_sig=False):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    global ENV
    ENV = eval_env()
    for defn in ast[:-1]:
        f_name = defn[1]
        f_v_is = defn[2]
        f_expr = defn[3]
        ENV.update({f_name: (f_v_is, f_expr)})
    l = {}
    ret, sig = evaluate(ast[-1], l, None)
    return (ret, sig) if return_sig else ret

# inspired by https://norvig.com/lispy.html
def evaluate(e, l, sig):
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
        test_value, sig = evaluate(test, l, sig)
        expr = (conseq if test_value else alt)
        return evaluate(expr, l, sig)
    # let statements
    elif e[0] == 'let':
        # get symbol
        symbol = e[1][0]
        # get value of e1
        value, sig = evaluate(e[1][1], l, sig)
        # evaluate e2 with value 
        return evaluate(e[2], {**l, symbol: value}, sig)
    # sample statement
    if e[0] == 'sample':
        dist, sig = evaluate(e[1], l, sig)
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        return dist.sample(), sig
    # observe statements
    # TODO: change this, maybe in this hw or for hw3
    if e[0] == 'observe':
        dist, sig = evaluate(e[1], l, sig) # get dist
        y, sig = evaluate(e[2], l, sig)    # get observed value
        # make sure it is a distribution object
        assert getattr(dist, '__module__', None).split('.')[:2] == ['torch', 'distributions']
        # TODO: do something with observed value
        return dist.sample(), sig
    # procedure call, either primitive or user-defined
    else:
        proc, sig = evaluate(e[0], l, sig)
        # primitives are functions
        if callable(proc):
            args = [None]*len(e[1:])
            for i, arg in enumerate(e[1:]):
                result, sig = evaluate(arg, l, sig)
                args[i] = result
            result = proc(*args)
            return result, sig
        # user defined functions are not
        else:
            # as written in algorithm 6
            v_is, e0 = proc 
            assert(len(v_is) == len(e[1:]))
            c_is = [None]*len(e[1:])
            for i, arg in enumerate(e[1:]):
                result, sig = evaluate(arg, l, sig)
                c_is[i] = result
            l_proc = dict(zip(v_is, c_is))
            return evaluate(e0, {**l, **l_proc}, sig)
    

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast, return_sig=True)
        print(ast)
        print(ret)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()

    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\n')
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast, return_sig=True)[0])
    