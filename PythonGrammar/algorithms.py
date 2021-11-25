#Fiboccina数列
def FibNum(n):
	if n==1:
		return 1
	elif n==2:
		return 1
	else:
		return FibNum(n-1)+FibNum(n-2)

for i in range(1,10):
	print(FibNum(i))

#Hanoi塔
def Hanoi(n,A,B,C):
	if n==1:
		print('Move disk from %s to %s'%(A,C))
	else:
		Hanoi(n-1,A,C,B)
		print('Move disk from %s to %s'%(A,C))
		Hanoi(n-1,B,A,C)
	return None


print('hello')