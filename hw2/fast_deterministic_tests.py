import torch
import pickle
from evaluation_based_sampling import evaluate_program
from tests import is_tol, run_prob_test,load_truth

infile = open('provided_det_tests_pkl', 'rb')
provided_tests = pickle.load(infile)
infile.close()

infile = open('extra_det_tests_pkl', 'rb')
extra_tests = pickle.load(infile)
infile.close()

all_tests = provided_tests + extra_tests

for test in all_tests:
	ast = test[0]
	truth = test[1]
	ret, sig = evaluate_program(ast, return_sig=True)
	print(ast)
	print(ret)
	try:
	    assert(is_tol(ret, truth))
	except AssertionError:
	    raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
    
	print('Test passed')
    
print('All deterministic tests passed')