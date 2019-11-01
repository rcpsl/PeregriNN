import pickle
fname = 'safety'
states = set(range(497))
with open(fname,'rb') as f:
    unsafe = set(pickle.load(f))

print(unsafe)
print(states-unsafe)
pass

set([48, 51, 14, 47, 16, 49, 18, 19, 84, 85, 15])