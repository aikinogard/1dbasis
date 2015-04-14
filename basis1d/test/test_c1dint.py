import basis1d.c1dints as c1dints

if __name__ == '__main__':
	print 'Mathematica result after normalization:'
	print '0.130432'
	print 'basis1d.c1dints.overlap1d result:'
	print c1dints.overlap1d(0.5,2,3,0.1,1,2.5)
