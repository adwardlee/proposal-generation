import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset_test import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

snippet_length = 6
os.environ["CUDA_VISIBLE_DEVICES"]="14,15"
# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','activitynet1.2','thumos14'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=1)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=1)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='flow_')
parser.add_argument('--save_path',type=str,default='fusion_activitynet1.2_train_result')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
### llj
elif args.dataset == 'activitynet1.2':
    num_class = 2
elif args.dataset == 'thumos14':
    num_class = 2
else:
    raise ValueError('Unknown dataset '+args.dataset)

####  llj
net = TSN(num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout, before_softmax=False)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ]))

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)

net.eval()

data_gen = data_loader

total_num = len(data_loader)
output = []

####   llj
def eval_video(video_data):
    i, data, video_id = video_data
    num_crop = args.test_crops

    scores_per_video = []

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
    #### llj
    for frame in data:
        #print('frame size',frame.size())
        ## batchsize*num_segment, rgb, h ,w
        input_var = torch.autograd.Variable(frame.view(-1, length, frame.size(1), frame.size(2)),
                                        volatile=True)
        ###batchsize * num_crop, 2
        rst = net(input_var).data.cpu().numpy().copy()
        ### batchsize , 2
        rst = rst.reshape((-1, args.test_segments, num_class)).mean(axis=1)
        #rst = rst.reshape((-1, num_crop, num_class)).mean(axis=1)
        #print('rst shape: ',rst.shape)
        #print('rst : ',rst[:,1])
        scores_per_video.extend(rst[:,1])
        #print('batch scores ',scores_per_video)

   # print('scores shape : ',np.shape(scores_per_video))
    return i, scores_per_video, video_id


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader)

### llj
for i in range(len(data_gen)):
    proc_start_time = time.time()
    if i >= max_num:
        break
    data, video_id, gt_seg = data_gen[i]
    num, scores, vid = eval_video((i, data, video_id))
    x = np.arange(len(scores) * snippet_length)#[i for i in range(len(scores))]
    #print(' x size: ',len(x))
    #print(' scores: ', scores)
    y = np.array([[j,j,j,j,j,j] for j in scores]) ### replicate snippet length times
    y = y.reshape(-1)
    #print(' y size : ',y.shape)
    #f = interp1d(x, y)

    plt.plot(x, y, '-',c='g')
    for num in range(len(gt_seg)):
        start = gt_seg[num][0]
        end = gt_seg[num][1]
        if end >= len(scores)*snippet_length:
            end = len(scores)*snippet_length -1
        tmp = np.array([0 if j < start or j > end else 1 for j in range(len(scores) * snippet_length)])
        #tmp[gt_seg[i]] = 1 - i * 0.02
        tmp.squeeze()
        #print('gt length', len(gt_seg))
        #print('tmp shape ',tmp.shape)
        plt.plot(x,tmp,c='b')
    #plt.plot(,'--',c='b')
    #plt.legend(['linear','true'], loc='best')
    real_vid = vid.split('/')[-1]
    plt.savefig(args.save_path + '/' + real_vid + '.jpg')
    plt.close()
    with open(args.save_path + '/' + real_vid +'.txt','w') as f:
        for score in scores:
            f.write(str(score))
            f.write('\n')
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)),flush=True)

# video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
#
# video_labels = [x[1] for x in output]
#
#
# cf = confusion_matrix(video_labels, video_pred).astype(float)
#
# cls_cnt = cf.sum(axis=1)
# cls_hit = np.diag(cf)
#
# cls_acc = cls_hit / cls_cnt
#
# print(cls_acc)
#
# print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
#
# if args.save_scores is not None:
#
#     # reorder before saving
#     name_list = [x.strip().split()[0] for x in open(args.test_list)]
#
#     order_dict = {e:i for i, e in enumerate(sorted(name_list))}
#
#     reorder_output = [None] * len(output)
#     reorder_label = [None] * len(output)
#
#     for i in range(len(output)):
#         idx = order_dict[name_list[i]]
#         reorder_output[idx] = output[i]
#         reorder_label[idx] = video_labels[i]
#
#     np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


