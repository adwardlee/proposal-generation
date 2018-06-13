import io
import requests

import matplotlib

import numpy as np
import pandas as pd
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gt_file = 'data/activitynet1.2_tag_val_prop.txt'
prop_file = 'data/real_props/final_step32_1_0.5_garma19-1_tao19-1_iou_0.8.txt'
tag_file = 'data/activitynet1.2_tag_val_proposal_list.txt'

#gt_file = 'data/thumos14/thumos14_tag_test_proposal_list.txt'
#prop_file = 'data/thumos14/thumos14_mytag_rgb_test_proposal_list.txt'
#tag_file = 'data/thumos14/thumos14_tag_test_proposal_list.txt'


def load_tag_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    out_dict = defaultdict(list)

    def parse_group(info):
        offset = 0
        vid = info[offset].split('/')[-1]
        offset += 1

        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split() for x in info[offset:offset+n_gt]]
        offset += n_gt

        n_pr = int(info[offset])
        offset += 1
        pr_boxes = [x.split()[3:] for x in info[offset:offset+n_pr]]


        return vid, pr_boxes

    for l in info_list:
        vid, pr_boxes = parse_group(l)
        out_dict[vid].append(pr_boxes)
    return out_dict

def load_groundtruth_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    out_dict = defaultdict(list)
    def parse_group(info):
        offset = 0
        vid = info[offset].split('/')[-1]
        offset += 1

        #n_frame = int(info[1] )
        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split()[1:] for x in info[offset:offset + n_gt]]
        offset += n_gt

        return vid, gt_boxes
    for l in info_list:
        vid, gt_boxes = parse_group(l)
        out_dict[vid].append(gt_boxes)
    return out_dict

def load_prop_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    out_dict = defaultdict(list)
    def parse_group(info):
        offset = 0
        vid = info[offset].split('/')[-1]
        offset += 1

        n_frame = int(info[1] )
        n_gt = int(info[2])
        offset = 3

        gt_boxes = [x.split() for x in info[offset:offset + n_gt]]
        offset += n_gt

        return vid, gt_boxes
    for l in info_list:
        vid, gt_boxes = parse_group(l)
        out_dict[vid].append(gt_boxes)
    return out_dict

""" Useful functions for the proposals evaluation.
"""
def segment_tiou(target_segments, test_segments, videoid):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        print('target_segment : ',target_segments.tolist())
        print('test_segment : ',test_segments.tolist())
        print('videoid : ',videoid)
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou

#### Average Recall vs Average number of proposals
def average_recall_vs_nr_proposals(proposals, ground_truth,
                                   tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Computes the average recall given an average number
        of proposals per video.

    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = np.array([x for x in proposals.keys()])
    total_prop_num = np.array([len(proposals[x][0]) for x in proposals.keys()])
    total_prop_num = np.sum(total_prop_num)

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        print('video id ', videoid,flush=True)
        this_video_proposals = np.array(proposals[videoid][0]).astype(np.int)
        this_video_ground_truth = np.array(ground_truth[videoid][0]).astype(np.int)

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals, videoid)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_lst = np.arange(1, 101) / 100.0
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        ## score_lst : one video one score, len(score_lst) = len(vided_lst)
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(score.shape[1] * pcn)
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(total_prop_num) / video_lst.shape[0])

    return recall, proposals_per_video

### recall vs iou threshold

def recall_vs_tiou_thresholds(proposals, ground_truth, nr_proposals=100,
                              tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    """ Computes recall at different tiou thresholds given a fixed
        average number of proposals per video.

    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    nr_proposals : int
        average number of proposals per video.
    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = np.array([x for x in proposals.keys()])
    total_prop_num = np.array([len(proposals[x][0]) for x in proposals.keys()])
    total_prop_num = np.sum(total_prop_num)

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:
        this_video_proposals = np.array(proposals[videoid][0]).astype(np.int)
        this_video_ground_truth = np.array(ground_truth[videoid][0]).astype(np.int)

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals, videoid)
        score_lst.append(tiou)

    # To obtain the average number of proposals, we need to define a
    # percentage of proposals to get per video.
    pcn = (video_lst.shape[0] * float(nr_proposals)) / total_prop_num

    # Computes recall at different tiou thresholds.
    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            # Get number of proposals at the fixed percentage of total retrieved.
            nr_proposals = int(score.shape[1] * pcn)
            # Find proposals that satisfies minimum tiou threhold.
            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()

    return recall, tiou_thresholds

### Retrieving DAPs proposal results (Thumos14)
# Retrieves and loads DAPs proposal results.

ground_truth = load_groundtruth_file(gt_file)
props = load_prop_file(prop_file)
#props = load_tag_file(prop_file)
props_tag = load_tag_file(tag_file)


# Computes average recall vs average number of proposals.
average_recall, average_nr_proposals = average_recall_vs_nr_proposals(props,
                                                                      ground_truth)

# Computes recall for different tiou thresholds at a fixed number of proposals.
recall, tiou_thresholds = recall_vs_tiou_thresholds(props, ground_truth,
                                                    nr_proposals=100)
tag_average_recall, tag_average_nr_proposals = average_recall_vs_nr_proposals(props_tag,
                                                                              ground_truth)

# Define plot style.
method = {'legend': 'TAG',
          'color': np.array([102,166,30]) / 255.0,
          'marker': None,
          'linewidth': 4,
          'linestyle': '-'}

method1 = {'legend': 'origin_TAG',
          'color': np.array([255,0,0]) / 255.0,
          'marker': None,
          'linewidth': 4,
          'linestyle': '-'}
fn_size = 14
plt.figure(num=None, figsize=(6, 5))

print(' props per video', average_nr_proposals)
print(' my largest recall : ',average_recall)
# Plots Average Recall vs Average number of proposals.
plt.semilogx(average_nr_proposals, average_recall,
             label=method['legend'],
             color=method['color'],
             linewidth=method['linewidth'],
             linestyle=str(method['linestyle']),
             marker=str(method['marker']))

plt.semilogx(tag_average_nr_proposals, tag_average_recall,
             label=method1['legend'],
             color=method1['color'],
             linewidth=method1['linewidth'],
             linestyle=str(method1['linestyle']),
             marker=str(method1['marker']))

plt.ylabel('Average Recall', fontsize=fn_size)
plt.xlabel('Average number of proposals', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 1.0])
plt.xlim([10**1, 10**4])
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
fig = plt.gcf()
plt.show()
fig.savefig(prop_file[:-4]+'.jpg')
plt.close()

# Plots recall at different tiou thresholds.
plt.plot(tiou_thresholds, recall,
         label=method['legend'],
         color=method['color'],
         linewidth=method['linewidth'],
         linestyle=str(method['linestyle']))

plt.grid(b=True, which="both")
plt.ylabel('Recall@100 proposals', fontsize=fn_size)
plt.xlabel('tIoU', fontsize=fn_size)
plt.ylim([0,1])
plt.xlim([0.1,1])
plt.xticks(np.arange(0, 1.2, 0.2))
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
fig = plt.gcf()
plt.show()
fig.savefig( prop_file[:-4]+'_tiou_' +'.jpg')
plt.close()