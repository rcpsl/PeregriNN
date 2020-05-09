import sys
sys.stdout = open('parsed.txt', 'w')
num_images = 100
timeout = 1200
results_file = 'result24_15.txt'

with open(results_file,'r') as f:
    lines = f.readlines()
    img_idx = 0
    prev_line = False
    for line in lines:
        if line == 'TIMEOUT!\n':
            print('-')
            img_idx += 1
            prev_line = True
        elif 'time' in line and prev_line is False:
            time = line.split(':')[-1].strip()
            print(time)
            img_idx += 1
        else:
            prev_line = False