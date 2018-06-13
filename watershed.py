import os
import sys
from multiprocessing import Pool
import numpy as np
import fnmatch
import argparse

parser = argparse.ArgumentParser(description="prop generation")
parser.add_argument('path', type=str, help='score path')
parser.add_argument('outfile', type=str, help='out file name')
parser.add_argument('--garma_left', type=int, default = 19)
parser.add_argument('--garma_right', type=int, default = 1)

parser.add_argument('--tao_left', type=int, default = 19)
parser.add_argument('--tao_right', type=int, default = 1)
parser.add_argument('--iou', type=int, default=0.8)

args = parser.parse_args()

path = args.path
#open_file = sys.argv[2]
garma_thres = [x*0.05 for x in range(args.garma_left, args.garma_right, -1)]
tao_thres = [x*0.05 for x in range(args.tao_left, args.tao_right,-1)]
processed_props = []

#out_file = open('fusion_activitynet1.2_train_result/result/train_garma18-1_tao18-1.txt','w')
out_file = open(args.outfile + str(args.garma_left)+'-'+
            str(args.garma_right)+'_tao'+str(args.tao_left) + '-'+str(args.tao_right) + '_iou_' + str(args.iou) + '.txt','w')
minimum_action_length = 5
snippet_length = 6

def consecutive_ones(a):
    isone = np.concatenate(([0],np.equal(a, 1).view(np.int8),[0]))
    absdiff = np.abs(np.diff(isone))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def nms(all_proposals, thresh = args.iou):
    if len(all_proposals) < 2:
        return np.array(all_proposals)
    proposals = np.array(all_proposals)
    pick = []
    x = proposals[:,0]
    y = proposals[:,1]
    scores = proposals[:,2]
    area = (y-x)
    idxs = scores.argsort()[::-1]
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x[i], x[idxs[1:]])
        yy1 = np.minimum(y[i], y[idxs[1:]])

        width = np.maximum(0.0, yy1 - xx1)

        overlap = width /(area[i] + area[idxs[1:]]-width)
        #print('overlap: ',overlap)
        inds = np.where(overlap <= thresh)[0]
        idxs = idxs[inds + 1]
    return proposals[pick]

def extract_segments(idx_file):
    idx, filename = idx_file

    #processed_props = []
    snippet_scores = []
    proposals = []
    #snippets = [0 for x in range(video_length[idx])]
    cnt = 0
    with open(os.path.join(path, filename), 'r') as f:
        for line in f:
            cnt += 1
            line = line.strip()
            snippet_scores.append(float(line))
    snippet_scores = np.array(snippet_scores)
    if  cnt < minimum_action_length: #### add llj
        proposals.append((0.0,1.0,1))
    else:
        for garma in garma_thres:
            initial_value = np.array([0 for x in range(cnt)])
            initial_value[np.where(snippet_scores>garma)] = 1
            ### get individual consecutive ones array
            fragments = consecutive_ones(initial_value)
            ## fragments return [start,end)
            for tao in tao_thres:
                for frag in fragments:
                    start = frag[0]
                    end = frag[1]
                    final_end = end
                    numerator = end -start
                    denominator = numerator
                    for add_on in range(end, cnt):
                        if initial_value[add_on] > 0:
                            numerator += 1
                        denominator += 1
                        if float(numerator)/denominator > tao:
                            final_end += 1
                        else:
                            break
                    if final_end - start < minimum_action_length:
                        continue
                    else:
                        segment_score = np.sum(snippet_scores[x] for x in range(start, final_end))/float(final_end-start)
                        proposals.append((float(start), float(final_end), segment_score))
                    ### reverse
                    r_start = frag[0]
                    r_end = frag[1]
                    r_final_start = r_start
                    numerator = r_end - r_start
                    denominator = numerator
                    for minus_on in range(r_start-1,-1,-1):
                        if initial_value[minus_on] > 0:
                            numerator += 1
                        denominator += 1
                        if float(numerator)/denominator > tao:
                            r_final_start -= 1
                        else:
                            break
                    if r_end - r_final_start < minimum_action_length:
                        continue
                    else:
                        segment_score = np.sum(snippet_scores[x] for x in range(r_final_start, r_end))/float(r_end-r_final_start)
                        proposals.append((float(r_final_start),float(r_end), segment_score))
    if len(proposals) == 0:
        proposals = np.array([[1,30,1]])
 

    proposals_length = [ x[1] - x[0] for x in proposals]
    max_length = max(proposals_length)
######## log(length)/log(max_length)
    proposals = [ [x[0],x[1], x[2] * (np.log(proposals_length[idx]+1.005)/np.log(max_length+1.005)) ] for idx, x in enumerate(proposals)]
    #proposals = [ [x[0],x[1], x[2] * (((proposals_length[idx]+1.005)**0.5)/((max_length+1.005)**0.5)) ] for idx, x in enumerate(proposals)]
########## sigmoid(length)    
    #proposals = [[x[0], x[1], 1/(1+np.exp(10-x[2]))] for x in proposals]
    proposals = nms(proposals)
    out_tmpl = "# {idx}\n{name}\n{fc}\n{num_prop}\n{props}"
    #props_dump = '\n'.join(['{} {}'.format(x[0], x[1]) for x in proposals]) + ('\n' if len(proposals)else '')

    props_dump = '\n'.join(['{} {}'.format(int(x[0]*snippet_length), int(x[1]*snippet_length)) for x in proposals]) + ('\n' if len(proposals) else '')
    processed_props.append(out_tmpl.format(idx=idx, name = filename[:-4], fc = snippet_length * cnt, num_prop=len(proposals), props=props_dump))
    
#out_file.write(filename[:-4])

 #   out_file.write('\n')
   # out_file.write(str(video_length[idx]))
    #out_file.write('\n')

    #for proposal in proposals:
    #        out_file.write(str(proposal[0]))
    #        out_file.write('\t')
    #        out_file.write(str(proposal[1]))
    #        out_file.write('\n')
    print('{} th video : {} done'.format(idx, filename),flush=True)
    #return processed_props



all_files = os.listdir(path)
files = []
for file_ in all_files:
    if fnmatch.fnmatch(file_,'*.txt'):
        files.append(file_)
print(' video nums are : ',len(files))



#workers = Pool(8)

#workers.map(extract_segments, enumerate(files))
for i, filename in enumerate(files):
    extract_segments((i, filename))
out_file.writelines(processed_props)
#workers = Pool(15)
#for i in range(len(garma_left)):
#    workers.map(generate,[(garma_left[i], tao_left[x]) for x in range(len(tao_left))])


