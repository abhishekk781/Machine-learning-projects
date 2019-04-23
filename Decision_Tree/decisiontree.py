import numpy as np 
import pandas as pd 

def entropy(col):
	uni = np.unique(col,return_counts=True)
	N = float(col.shape[0])

	ent=0.0
	for ix in uni[1]:
		p = ix/N
		ent += (-1*p*np.log2(p))

	return ent


def divide_data(x_data,fkey,fval):
	x_right = pd.DataFrame([],coloums=x_data.coloums)
	x_left = pd.DataFrame([],coloums=x_data.coloums)
	for i in range(x_data.shape[0]):
		val = x_data[fkey].loc[i]
		if val>fval:
			x_right = x_right.append(x_data.loc[i])
		else:
			x_left = x_left.append(x_data.loc[i])

	return x_left,x_right


def informationGain(x_data,fkey,fval):
	x_left,x_right = divide_data(x_data,fkey,fval)

	l = float(x_left.shape[0]/x_data.shape[0])
	r = float(x_right.shape[0]/x_data.shape[0])

	if x_left.shape[0]==0 or x_right.shape[0]==0:
		return -100000000

	i_gain = entropy(x_data.Survived) - (l*entropy(x_left.Survived) + r*entropy(x_right.Survived))
	return i_gain

lst = np.array([1,1,0,1,0,0])
ans = entropy(lst)
print(ans)