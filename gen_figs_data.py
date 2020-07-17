import pandas as pd
import matplotlib.pyplot as plt

df  = pd.read_csv(r'results/Acas_results.csv')
n_rows = len(df)
counter = 0
thresh = 10
acc_time = 0
acc_proved = 0
# out_file = 'Acas_fig_data.txt'
# f = open(out_file, "a")
x=[]
y=[]

for row_idx in range(n_rows):
    if(df['result'][row_idx] !='timeout'):
        acc_proved +=1
    acc_time += df['time'][row_idx]
    if(acc_time - counter >= thresh):
        # f.write("%d,%d\n"%(int(acc_time),acc_proved))
        x.append(acc_time)
        y.append(acc_proved)
        counter = acc_time
# f.close()   

plt.xscale('log')
plt.ylim(0, 186)
plt.xlabel('Timeout(sec)')
plt.ylabel('Number of verified properties')
plt.grid(ls='-')
# plt.xlim(right = 100)
plt.plot(x,y)
plt.savefig('results/acas.png')
plt.clf()

###### MNIST########

df  = pd.read_csv(r'results/mnist_results.csv')


idx = 0
n_images = 50
for network in range(3):
    for eps in [0.01, 0.02, 0.03]:
        counter = 0
        thresh = 2
        acc_time = 0
        acc_proved = 0   
        x = []
        y = []
        plt.ylim(0, 60)
        plt.xscale('log')
        plt.grid(ls='-')
        for _ in range(n_images):
            if(df['result'][idx] !='timeout'):
                acc_proved +=1
            acc_time += df['time'][idx]
            if(acc_time - counter >= thresh or idx%49 ==0):
                # f.write("%d,%d\n"%(int(acc_time),acc_proved))
                x.append(acc_time)
                y.append(acc_proved)
                counter = acc_time
            idx +=1
        plt.plot(x,y,label=str(eps))
    plt.legend(loc="upper right")
    plt.savefig('results/mnist%d.png'%network)

    plt.clf()