import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

import math
from transforms import *
from ops.io import load_file ##llj
import sys

snippet_length = 6
sampling_interval =6
fg_per_video = 1
bg_per_video = 1
####how many negatives vs positives?

class SSNInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None,
                 best_iou=None, overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps

        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self


    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1

class VideoRecord:
    def __init__(self, gt_record, num_segments, new_length):
        self._data = gt_record

        frame_count = int(self._data[1])
        
        gt_list = []
        for x in self._data[2]:
            gt_list.append([int(x[1]),int(x[2])])
        gt_list.sort(key=lambda x: x[0])
        # build instance record
        self.gt = [
            SSNInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]

        self.overlapped_gt = [gt_list[0]]
        for current in gt_list:
            previous = self.overlapped_gt[-1]
            if current[0] < previous[1]:
                previous[1] = max(current[1],previous[1])
            else:
                self.overlapped_gt.append(current)
        

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))
        self.num_segments = num_segments
        self.new_length = new_length


    @property
    def id(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    ##### get the ground truth action snippet ###########llj
    def get_fg_negatives(self):

        fg = [p for p in self.gt]
        true_fg = []
        true_bg = []
        for x in range(len(fg)):
            if (fg[x].end_frame - fg[x].start_frame) > self.num_segments *self.new_length:
                true_fg.append(SSNInstance(int(fg[x].start_frame),int(fg[x].end_frame),self.num_frames, label = 1, best_iou=1.0))
        bg_instance = []

        ### get bg instance ####
        overlapp_fg = self.overlapped_gt
        if  (overlapp_fg[0][0] -1) -1 > self.num_segments * self.new_length:### llj
            bg_instance.append([1, overlapp_fg[0][0] - 1])
        for length in range(len(overlapp_fg)-1):
            if (overlapp_fg[length+1][0] - 1) - (overlapp_fg[length][1]+1) > self.num_segments * self.new_length:####llj
                bg_instance.append([overlapp_fg[length][1] + 1 , overlapp_fg[length+1][0] -1])
        if self.num_frames - (overlapp_fg[-1][1]+1) > self.num_segments * self.new_length:
            bg_instance.append([overlapp_fg[-1][1] + 1, self.num_frames])

        for y in bg_instance:
            if (y[1] - y[0]) > self.num_segments *self.new_length:
                true_bg.append(SSNInstance(int(y[0]),int(y[1]),self.num_frames, label = 0, best_iou=1.0))

        return true_fg, true_bg


#### change fg_per_video and bg_per_video
class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=5, new_length=1, modality='RGB',
                 image_tmpl='image_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, verbose=True,
                 video_centric=True,fg_per_video = fg_per_video, bg_per_video = bg_per_video):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.verbose = verbose
        self.video_centric = video_centric
        self.sampling_interval = sampling_interval
        self.snippet_length = snippet_length
        self.fg_per_video = fg_per_video
        self.bg_per_video = bg_per_video

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

### get the 5frame-length foreground and background
    def _parse_list(self, stats = None):
        box_info = load_file(self.list_file)

        self.video_list = [VideoRecord(p, self.num_segments, self.new_length) for p in box_info]

        self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        self.video_dict = {v.id: v for v in self.video_list} ### video id list

        self.fg_pool = []
        self.bg_pool = []

############ get lowest and highest frames number in the dataset
        lowest = 100000
        highest = 0

        for v in self.video_list:
            fg_list, bg_list = v.get_fg_negatives()
            self.fg_pool.extend([v.id, prop] for prop in fg_list)
            self.bg_pool.extend([v.id, prop] for prop in bg_list)
            num_frames = v.num_frames
            if num_frames > highest:
                highest = num_frames
            if num_frames < lowest:
                lowest = num_frames
        #### test fg_pool bg_pool
        #print('positive : \n')
        #for i, x in enumerate(self.fg_pool):
        #    print('num: ',i , ' vid: ',x[0] ,' start: ',x[1].start_frame, ' end: ', x[1].end_frame)
        #print('negative : \n')
        #for i, x in enumerate(self.bg_pool):
        #    print('num : ', i, ' vid: ',x[0],' start: ',x[1].start_frame, ' end : ',x[1].end_frame)

        if self.verbose:
            print("""

                    SSNDataset: video file {prop_file} parsed.

                    There are {pnum} usable snippet from {vnum} videos.
                    {fnum} foreground proposals
                    {bnum} background_proposals

                    Sampling config:
                    FG/BG: {fr}/{br}
                    Video Centric: {vc}
                    lowest frame num: {low}
                    highest frame num: {high}


                    """.format(prop_file=self.list_file,
                               pnum=len(self.fg_pool) + len(self.bg_pool),
                               fnum=len(self.fg_pool), bnum=len(self.bg_pool),
                               fr=self.fg_per_video, br=self.bg_per_video,
                               vnum=len(self.video_dict),
                               vc=self.video_centric, low=lowest, high=highest))
        else:
            print("""
                                SSNDataset: Proposal file {prop_file} parsed.   
                    """.format(prop_file=self.list_file))

    def _video_centric_sampling(self, video):
        fg, bg = video.get_fg_negatives()

        def sample_video_proposals(proposal_type, video_id, video_pool, requested_num, dataset_pool):
            if len(video_pool) == 0:
                # if there is nothing in the video pool, go fetch from the dataset pool
                return [(dataset_pool[x], proposal_type) for x in np.random.choice(len(dataset_pool), requested_num, replace=False)]
            else:
                replicate = len(video_pool) < requested_num
                idx = np.random.choice(len(video_pool), requested_num, replace=replicate)
                return [((video_id, video_pool[x]), proposal_type) for x in idx]

        out_props = []
        ########## proposal type : fg = 1 , bg = 0   ### llj
        out_props.extend(sample_video_proposals(1, video.id, fg, self.fg_per_video, self.fg_pool))  # sample foreground
        out_props.extend(sample_video_proposals(0, video.id, bg, self.bg_per_video, self.bg_pool))  # sample background

        return out_props

#### sampling snippet from all the videos
    def _random_sampling(self):
        out_props = []

        ########## proposal type : fg = 1 , bg = 0   ### llj
        out_props.extend([(x, 1) for x in np.random.choice(self.fg_pool, self.fg_per_video, replace=False)])
        out_props.extend([(x, 0) for x in np.random.choice(self.bg_pool, self.bg_per_video, replace=False)])

        return out_props


    def _sample_ssn_indices(self, prop):
        start_frame = prop.start_frame
        end_frame = prop.end_frame

        duration = end_frame - start_frame
        assert duration != 0, (prop.start_frame, prop.end_frame)

        average_duration = (duration -self.new_length+1)//self.num_segments

        #### llj
        if average_duration > 0:
            offsets = start_frame + np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif duration > self.num_segments:
            offsets = start_frame + np.sort(randint(duration - self.new_length + 1, size=self.num_segments))
        else:
            offsets = start_frame + np.zeros((self.num_segments,))

        #offsets.sort()

        return offsets+1

    def _load_prop_data(self, prop):

        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames


        # sample segment indices
        #### num_segments index
        prop_indices = self._sample_ssn_indices(prop[0][1])

        # turn prop into standard format
        print('vid : ',prop[0][0])
        print('prop indices: ', prop_indices)
        print('label : ',prop[1])
        # get label
        if prop[1] == 1:
            label = 1 ### foreground
        elif prop[1] == 0:
            label = 0  # background
        else:
            raise ValueError()
        frames = []
        if self.modality == 'RGB':
            for idx, seg_ind in enumerate(prop_indices):
                p = int(seg_ind)
                for x in range(self.new_length):
                    frames.extend(self._load_image(prop[0][0], min(frame_cnt, p+x)))
        elif self.modality == 'RGBDiff' or self.modality == 'Flow':
            for idx, seg_ind in enumerate(prop_indices):
                p = int(seg_ind)
                for x in range(self.new_length):
                    frames.extend(self._load_image(prop[0][0], min(frame_cnt, p +x)))

        return frames, label, prop[1]

    def get_training_data(self, index):
        if self.video_centric:
            video = self.video_list[index]
            ### (fg_per_video + bg_per_video) snippet
            props = self._video_centric_sampling(video)
        else:
            props = self._random_sampling()

        out_frames = []

        out_prop_labels = []
        for idx, p in enumerate(props):
            ## num_seg*3*w*h, 1, 1
            prop_frames, prop_label, prop_type = self._load_prop_data(p)
            sys.stdout.flush()
            ##### (num_seg*3) * w*h
            processed_frames = self.transform(prop_frames)
            out_frames.append(processed_frames)
            out_prop_labels.append(prop_label)
        ####### (fg_per_video + bg_per_video) labels
        out_prop_labels = torch.from_numpy(np.array(out_prop_labels))
        ###### (fg_per_video + bg_per_video)*num_seg*3, w, h
        out_frames = torch.cat(out_frames)
        return out_frames, out_prop_labels

    def get_testing_data(self,index, batch_size = 4):
        video = self.video_list[index]
        video_id = video.id
        fg = [p for p in video.gt]
        gt_range = [[fg[x].start_frame, fg[x].end_frame] for x in range(len(fg))]
        #### sampling the snippets
        frame_cnt = video.num_frames
        frame_ticks = np.arange(0, frame_cnt, self.snippet_length) + 1
        num_sampled_snippet = len(frame_ticks)

        for num in range(int(math.floor((video.num_frames-1)//self.sampling_interval))):
            start_frame = num * self.sampling_interval + 1
            end_frame = num * self.sampling_interval + snippet_length

        def frame_gen(batchsize):
            frames = []
            cnt = 0
            for idx, seg_ind in enumerate(frame_ticks):
                p = int(seg_ind)
                offsets = []
                #### llj
                start_frame = p
                end_frame = p + self.snippet_length -1
                candidate_frames = [x for x in range(start_frame, end_frame + 1)]
                offsets.extend(x for x in sorted(random.sample(candidate_frames, self.num_segments)))
                if self.modality == "RGB":
                    for frame_idx in offsets:
                        for x in range(self.new_length):
                            frames.extend(self._load_image(video_id, min(frame_cnt, frame_idx)))
                elif self.modality == "RGBDiff" or self.modality == "Flow":
                    for x in range(self.new_length):
                        frames.extend(self._load_image(video_id,min(frame_cnt, start_frame + x)))
                cnt += 1

                # for x in range(self.new_length):
                #     frames.extend(self._load_image(video_id, min(frame_cnt, p + x)))
                # cnt += 1

                if cnt % batchsize == 0:
                    frames = self.transform(frames)
                    yield frames
                    frames = []

            if len(frames):
                frames = self.transform(frames)
                yield frames
        return frame_gen(batch_size), video_id, gt_range


    # def get_test_data(self, video, test_interval, gen_batchsize=4):
    #     props = video.gt
    #     video_id = video.id
    #     frame_cnt = video.num_frames
    #     frame_ticks = np.arange(0, frame_cnt - self.new_length, test_interval, dtype=np.int) + 1
    #
    #     num_sampled_frames = len(frame_ticks)
    #
    #     # avoid empty proposal list
    #     if len(props) == 0:
    #         props.append(SSNInstance(0, frame_cnt - 1, frame_cnt))
    #
    #     # process proposals to subsampled sequences
    #     rel_prop_list = []
    #     proposal_tick_list = []
    #     for proposal in props:
    #         rel_prop = proposal.start_frame / frame_cnt, proposal.end_frame / frame_cnt
    #         rel_duration = rel_prop[1] - rel_prop[0]
    #         rel_starting_duration = rel_duration * self.starting_ratio
    #         rel_ending_duration = rel_duration * self.ending_ratio
    #         rel_starting = rel_prop[0] - rel_starting_duration
    #         rel_ending = rel_prop[1] + rel_ending_duration
    #
    #         real_rel_starting = max(0.0, rel_starting)
    #         real_rel_ending = min(1.0, rel_ending)
    #
    #
    #         proposal_ticks = int(real_rel_starting * num_sampled_frames), int(rel_prop[0] * num_sampled_frames), \
    #                          int(rel_prop[1] * num_sampled_frames), int(real_rel_ending * num_sampled_frames)
    #
    #         rel_prop_list.append(rel_prop)
    #         proposal_tick_list.append(proposal_ticks)
    #
    #     # load frames
    #     # Since there are many frames for each video during testing, instead of returning the read frames,
    #     # we return a generator which gives the frames in small batches, this lower the memory burden
    #     # and runtime overhead. Usually setting batchsize=4 would fit most cases.
    #     def frame_gen(batchsize):
    #         frames = []
    #         cnt = 0
    #         for idx, seg_ind in enumerate(frame_ticks):
    #             p = int(seg_ind)
    #             for x in range(self.new_length):
    #                 frames.extend(self._load_image(video_id, min(frame_cnt, p+x)))
    #             cnt += 1
    #
    #             if cnt % batchsize == 0:
    #                 frames = self.transform(frames)
    #                 yield frames
    #                 frames = []
    #
    #         if len(frames):
    #             frames = self.transform(frames)
    #             yield frames
    #
    #     return frame_gen(gen_batchsize), len(frame_ticks), torch.from_numpy(np.array(rel_prop_list)), \
    #            torch.from_numpy(np.array(proposal_tick_list))


############ original ############
    # def _sample_indices(self, record):
    #     """
    #
    #     :param record: VideoRecord
    #     :return: list
    #     """
    #
    #     average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
    #     if average_duration > 0:
    #         offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
    #     elif record.num_frames > self.num_segments:
    #         offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
    #     else:
    #         offsets = np.zeros((self.num_segments,))
    #     return offsets + 1
    #
    # def _get_val_indices(self, record):
    #     if record.num_frames > self.num_segments + self.new_length - 1:
    #         tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
    #         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
    #     else:
    #         offsets = np.zeros((self.num_segments,))
    #     return offsets + 1

    # def _get_test_indices(self, record):
    #
    #     tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
    #
    #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
    #
    #     return offsets + 1

    def __getitem__(self, index):
        #record = self.video_list[index]
        real_index = index % len(self.video_list)
        if not self.test_mode:
            return self.get_training_data(real_index)
            #### llj
            #segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            return self.get_testing_data(real_index) ###llj
            #segment_indices = self._get_test_indices(record)

        #return self.get(record, segment_indices)

    def __len__(self):
        return len(self.video_list) #* 2
