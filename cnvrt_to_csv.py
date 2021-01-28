import pandas as pd

read  = pd.read_csv(r'Acas_results.txt')
read.columns = ['network','prop','result','time']
read.to_csv (r'results/Acas_results.csv', index=None)

read  = pd.read_csv(r'mnist_results.txt')
read.columns = ['network','eps','result','time']
read.to_csv (r'results/mnist_results.csv', index=None)