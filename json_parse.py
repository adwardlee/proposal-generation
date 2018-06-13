import json
from collections import defaultdict
import glob
import fnmatch
import os
import argparse

parser = argparse.ArgumentParser(
    description="Generate proposal list to be used for training")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'thumos14'])
parser.add_argument('frame_path', type=str)

args = parser.parse_args()

if args.dataset == 'activitynet1.2':
    json_file = 'activity_net.v1-2.min.json'
    key_func = lambda x: x[-11:]

elif args.dataset == 'activitynet1.3':
    json_file = 'activity_net.v1-3.min.json'
    key_func = lambda x: x[-11:]
elif args.dataset == 'thumos14':
    json_file = 'thumos14.json'
    key_func = lambda x: x.split('/')[-1]
else:
    raise ValueError("unknown dataset : {}".format(args.dataset))

out_list_tmpl = 'data/{}_gt_list.txt'

### parse dir to load frames
def parse_directory(path, key_func=lambda x: x[-11:],
                    rgb_prefix='image_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = key_func(f)

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (f, all_cnt[0], x_cnt)

    print('frame folder analysis done')
    return frame_dict

def parse_json(json_file):
    class_name = set()
    data = json.load(open(json_file))
    class_ind = defaultdict(dict)
    for i in range(len(data['taxonomy'])):
        class_ind[data['taxonomy'][i]['nodeName']] = data['taxonomy'][i]['nodeId']

    train_dict = defaultdict(dict)
    val_dict = defaultdict(dict)

    for key in data['database']:
        if data['database'][key]['subset'] == 'training':
            train_dict[key]['segment'] = []
            train_dict[key]['num_segment'] = 0
            for j in range(len(data['database'][key]['annotations'])):
                label1 = data['database'][key]['annotations'][j]['label']
                class_name.add(label1)
                label = class_ind[label1]
                train_dict[key]['segment'].append([label,
                                                   data['database'][key]['annotations'][j]['segment'][0],
                                                   data['database'][key]['annotations'][j]['segment'][1] ])
                train_dict[key]['num_segment'] += 1
            train_dict[key]['duration'] = data['database'][key]['duration']
        if data['database'][key]['subset'] == 'validation':
            val_dict[key]['segment'] = []
            val_dict[key]['num_segment'] = 0
            for j in range(len(data['database'][key]['annotations'])):
                label1 = data['database'][key]['annotations'][j]['label']
                label = class_ind[label1]
                class_name.add(label1)
                val_dict[key]['segment'].append([label,
                                                data['database'][key]['annotations'][j]['segment'][0],
                                                data['database'][key]['annotations'][j]['segment'][1]])
                val_dict[key]['num_segment'] += 1
            val_dict[key]['duration'] = data['database'][key]['duration']
        else:
           continue
    with open('class_name.txt','w') as f:
        for x in class_name:
            f.write(x)
    return train_dict, val_dict



def generate_list(data_dict, frame_dict, out_list_name):
    missing_video_number = 0
    processed_proposal_list = []
    for idx, vid in enumerate(data_dict.keys()):
        if vid not in frame_dict:
            missing_video_number += 1
            continue
        frame_info = frame_dict[vid]
        frame_cnt = frame_info[1]
        frame_path = frame_info[0]

        gt_num = data_dict[vid]['num_segment']
        duration = data_dict[vid]['duration']
        gt = [[int(x[0]), int((float(x[1])/duration) * frame_cnt), int((float(x[2])/duration) * frame_cnt)] for x in data_dict[vid]['segment']]

        out_tmpl = "# {idx}\n{path}\n{fc}\n1\n{num_gt}\n{gt}"

        gt_dump = '\n'.join(['{} {:d} {:d}'.format(*x) for x in gt]) + ('\n' if len(gt) else '')

        processed_proposal_list.append(out_tmpl.format(
            idx=idx, path=frame_path, fc=frame_cnt,
            num_gt=len(gt), gt=gt_dump
        ))

    print('missing video numbers : ',missing_video_number)
    open(out_list_name, 'w').writelines(processed_proposal_list)

frame_dict = parse_directory(args.frame_path, key_func=key_func)

train_dict, val_dict = parse_json(json_file)

generate_list(train_dict, frame_dict, out_list_tmpl.format('train'))
generate_list(val_dict, frame_dict, out_list_tmpl.format('val'))

print("train/val lists for dataset {} are ready for training.".format(args.dataset))