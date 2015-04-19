from basis1d.tools import dict2str
import numpy as np

def compare_dictionaries(dict1, dict2):
	"""
	compare dicts from http://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
	"""
	if dict1 == None or dict2 == None:
		return False

	if type(dict1) is not dict or type(dict2) is not dict:
		return False

	shared_keys = set(dict2.keys()) & set(dict2.keys())

	if not ( len(shared_keys) == len(dict1.keys()) and len(shared_keys) == len(dict2.keys())):
		return False

	dicts_are_equal = True
	for key in dict1.keys():
		if type(dict1[key]) is dict:
			dicts_are_equal = dicts_are_equal and compare_dictionaries(dict1[key],dict2[key])
		else:
			if type(dict1[key]).__module__=='numpy':
				dicts_are_equal = dicts_are_equal and np.allclose(dict1[key],dict2[key])
			else:
				dicts_are_equal = dicts_are_equal and (dict1[key] == dict2[key])

	return dicts_are_equal


a = {'name':'test',4:6,'order':1.,'subdict':{'s':0,'p':1},'np.array':np.arange(20.).reshape((4,5)),'lists':[[1,2,3],[5.,4.]]}
stra = dict2str(a)
print stra
# copy paste print stra as b
b=\
{
	4:
		6,
	'name':
		'test',
	'lists':
		[[1, 2, 3], [5.0, 4.0]],
	'subdict':
		{
			'p':
				1,
			's':
				0,
		},
	'np.array':
		np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0]]),
	'order':
		1.0,
}

print 'test whether they equal:'
print compare_dictionaries(a, b)
