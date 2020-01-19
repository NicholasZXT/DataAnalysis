import copy

L=['a','b',['c','d']]
L_ref=L
L_shallow=L.copy()
L_deep=copy.deepcopy(L)
id(L)
id(L_ref)
id(L_shallow)
id(L_deep)


L=['a','b',['c','d']]
D={'a':1,'b':2}
T='a','b'

def changeList(L):
	L.append('add')
	return(L)

def changeDict(D):
	D['c']='add'
	return(D)

def changeTuple(T):
	T=T,'add'
	return(T)

ham=[0]
def change(egg):
	egg.append(1)
	egg=[2,3]

