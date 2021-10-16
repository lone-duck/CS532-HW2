from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import eval_env
import torch


def evaluate_program(ast, return_sig=False):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    env = eval_env()
    ret, sig = evaluate(ast, env)
    return (ret, sig) if return_sig else ret

def evaluate(e, env, sig=None):
    # variable reference
    if isinstance(e, str):        
        return env[e], sig
    # constant number
    elif isinstance(e, (int, float)):   
        return torch.tensor(float(e)), sig
    # root of tree
    # THIS MUST BE FIXED TO ACCOUNT FOR DEFNs!!!!
    elif isinstance(e, list) and len(e) == 1:  
        return evaluate(e[0], env)
    # STUFF NEEDS TO GO HERE
    # procedure call
    else:
        proc, sig = evaluate(e[0], env)
        args = [evaluate(arg, env)[0] for arg in e[1:]]
        result, sig = proc(*args), sig
        return result, sig


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print(ast)
        ret, sig = evaluate_program(ast, return_sig=True)
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
    """
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])
    """