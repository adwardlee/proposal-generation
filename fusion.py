import numpy as np
import sys
import os
import scipy.signal
import fnmatch

filter_size = 32
#out_path = 'thumos14_scores/fusion_val_step32_1_0.5'

out_path = sys.argv[1]
#out_path = 'diff_fusion/real_step1_1_0.5'
mismatch_files = 0
rgb_weight = 1
flow_weight = 0.5
#rgb_path = 'thumos14_scores/rgb_val'
rgb_path = sys.argv[2]
#flow_path = 'thumos14_scores/flow_val'
flow_path = sys.argv[3]
rgb_files = os.listdir(rgb_path)
flow_files = os.listdir(flow_path)
files = []
for file_ in rgb_files:
    if fnmatch.fnmatch(file_,'*.txt'):
        files.append(file_)
print(' video nums are : ',len(files))

files1 = []
for file_ in flow_files:
    if fnmatch.fnmatch(file_,'*.txt'):
        if file_ not in files:
            print(' ',file_)
            raise Exception(' not in rgb dir: ')

for file_ in files:
    rgb_scores = []
    flow_scores = []
    scores = []
    with open(rgb_path+'/'+ file_,'r') as f:
        for line in f:
            line = line.strip()
            rgb_scores.append(float(line))
    with open(flow_path + '/' + file_,'r') as f:
        for line in f:
            line = line.strip()
            flow_scores.append(float(line))
    with open(out_path + '/' + file_,'w') as f:
        if len(rgb_scores) != len(flow_scores):
            print(' ',file_)
            #print(' flow length less than rgb')
            sys.stdout.flush()
            mismatch_files += 1
            #raise Exception(' flow rgb length not equal')
        
        for i in range(len(flow_scores)):
            scores.append((rgb_scores[i] * rgb_weight + flow_scores[i] * flow_weight)/(rgb_weight + flow_weight))
        if len(scores) > filter_size:
            #scores_hat = scipy.signal.savgol_filter(scores,filter_size,3)  ### savgol filter

            #scores_hat = scores
            scores_hat = np.convolve(scores,np.ones(filter_size)/filter_size,mode='same')    
        else:
            #print('filename : ',file_)
            scores_hat = np.convolve(scores,np.ones(len(scores)//2)/(len(scores)//2), mode='same')
        for score in scores_hat:
            if score < 0:
                score = 0
            f.write(str(score))
            f.write('\n')

#print('mis match file num: ',mismatch_files)
