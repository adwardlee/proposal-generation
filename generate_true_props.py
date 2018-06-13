import os
import sys
import numpy as np
import fnmatch
from collections import defaultdict

def load_groundtruth_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    out_dict = defaultdict(list)
    def parse_group(info):
        offset = 0
        vid = info[offset]
        offset += 1

        n_frame = int(info[1] )
        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split() for x in info[offset:offset + n_gt]]
        offset += n_gt

        return vid, n_frame, gt_boxes
    for l in info_list:
        vid, n_frame, gt_boxes = parse_group(l)
        out_dict[vid].extend([n_frame,gt_boxes])
    return out_dict

def load_prop_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(info):
        offset = 0
        vid = info[offset]
        offset += 1

        n_frame = int(info[1] )
        n_gt = int(info[2])
        offset = 3

        gt_boxes = [x.split() for x in info[offset:offset + n_gt]]
        offset += n_gt

        return vid, n_frame, gt_boxes

    return [parse_group(l) for l in info_list]

    # for l in info_list:
    #     vid, gt_boxes = parse_group(l)
    #     out_dict[vid].append(gt_boxes)
    # return out_dict


def compute_iou_overlap(gts, prop, max_num):
    max_iou = 0
    max_overlap = 0
    class_num = 0
    prop_left = int(prop[0])
    prop_right = int(prop[1])
    if prop_right >max_num:
        prop_right = max_num
    for gt in gts:
        gt_start = gt[1]
        gt_end = gt[2]
        g_class = gt[0]
        max_length = max(gt_end,prop_right) - min(gt_start, prop_left)
        intersec = min(gt_end, prop_right) - max(gt_start,prop_left)
        iou = intersec/float(max_length)
        overlap = intersec/float(prop_right-prop_left)
        if iou > max_iou:
            max_iou = iou
            class_num = g_class
        if overlap > max_overlap:
            max_overlap = overlap
    return [class_num, max_iou, max_overlap, prop_left, prop_right]

def generate_prop_list(gt_file, prop_file, out_list_name):
    gt_dict = load_groundtruth_file(gt_file)
    prop_dict = load_prop_file(prop_file)
    processed_proposal_list = []
    for idx, prop in enumerate(prop_dict):
        vid = prop[0]
        props = prop[2]
        for x in gt_dict.keys():
            if vid in x:
                gt_box = gt_dict[x][1]
                frame_cnt = gt_dict[x][0] -1
                frame_path = x
                gt = [ [int(y[0]), int(y[1]), int(y[2])] for y in gt_box]
                prop = [ compute_iou_overlap(gt, y, frame_cnt) for y in props]

                out_tmpl = "# {idx}\n{path}\n{fc}\n1\n{num_gt}\n{gt}{num_prop}\n{prop}"

                gt_dump = '\n'.join(['{} {:d} {:d}'.format(*x) for x in gt]) + ('\n' if len(gt) else '')
                prop_dump = '\n'.join(['{} {:.04f} {:.04f} {:d} {:d}'.format(*x) for x in prop]) + (
                    '\n' if len(prop) else '')

                processed_proposal_list.append(out_tmpl.format(
                    idx=idx, path=frame_path, fc=frame_cnt,
                    num_gt=len(gt), gt=gt_dump,
                    num_prop=len(prop), prop=prop_dump
                ))

    open(out_list_name, 'w').writelines(processed_proposal_list)
    print('successfully generate files')

gt_file = 'data/gt/activitynet1.2_flow_val_gt_list.txt'
prop_file = '1_0.5_step64_garma18-1_tao18-1_iou08_fusion.txt'
out_name = 'my_prop_flow_val.txt'

generate_prop_list(gt_file, prop_file, out_name)