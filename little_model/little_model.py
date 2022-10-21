# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import copy
import logging

import cv2
import numpy as np
import os

from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.joint_inference import JointInference

from interface import Estimator


LOG = logging.getLogger(__name__)

class_names = ['person', 'helmet', 'helmet_on', 'helmet_off']
all_output_path = Context.get_parameters(
    'all_examples_inference_output'
)
hard_example_edge_output_path = Context.get_parameters(
    'hard_example_edge_inference_output'
)
hard_example_cloud_output_path = Context.get_parameters(
    'hard_example_cloud_inference_output'
)

FileOps.clean_folder([
    all_output_path,
    hard_example_cloud_output_path,
    hard_example_edge_output_path
], clean=False)


def draw_boxes(img, bboxes, colors, text_thickness, box_thickness):
    img_copy = copy.deepcopy(img)

    line_type = 2
    #  get color code
    colors = colors.split(",")
    colors_code = []
    for color in colors:
        if color == 'green':
            colors_code.append((0, 255, 0))
        elif color == 'blue':
            colors_code.append((255, 0, 0))
        elif color == 'yellow':
            colors_code.append((0, 255, 255))
        else:
            colors_code.append((0, 0, 255))

    label_dict = {i: label for i, label in enumerate(class_names)}

    for bbox in bboxes:
        if float("inf") in bbox or float("-inf") in bbox:
            continue
        label = int(bbox[5])
        score = "%.2f" % round(bbox[4], 2)
        text = label_dict.get(label) + ":" + score
        p1 = (int(bbox[1]), int(bbox[0]))
        p2 = (int(bbox[3]), int(bbox[2]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        try:
            cv2.rectangle(img_copy, p1[::-1], p2[::-1], colors_code[label],
                          box_thickness)
            cv2.putText(img_copy, text, (p1[1], p1[0] + 20 * (label + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                        text_thickness, line_type)
        except TypeError as err:
            # error message from pyopencv,  cv2.circle only can accept centre
            # coordinates precision up to float32. If the coordinates are in
            # float64, it will throw this error.
            LOG.warning(f"Draw box fail: {err}")
    return img_copy


def output_deal(
        final_result,
        is_hard_example,
        cloud_result,
        edge_result,
        nframe,
        img_rgb
):
    # save and show image
    img_rgb = np.array(img_rgb)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    collaboration_frame = draw_boxes(img_rgb, final_result,
                                     colors="green,blue,yellow,red",
                                     text_thickness=None,
                                     box_thickness=None)

    cv2.imwrite(f"{all_output_path}/{nframe}.jpeg", collaboration_frame)

    # save hard example image to dir
    if not is_hard_example:
        return

    if cloud_result is not None:
        cv2.imwrite(f"{hard_example_cloud_output_path}/{nframe}.jpeg",
                    collaboration_frame)
    edge_collaboration_frame = draw_boxes(
        img_rgb,
        edge_result,
        colors="green,blue,yellow,red",
        text_thickness=None,
        box_thickness=None)
    cv2.imwrite(f"{hard_example_edge_output_path}/{nframe}.jpeg",
                edge_collaboration_frame)

HIST_NB_BINS = 32
HIST_DIFF_THRESHOLD = 0.1

def get_Hist(frame):
    nb_channels = frame.shape[-1]
    hist = np.zeros((HIST_NB_BINS * nb_channels, 1), dtype='float32')
    for i in range(nb_channels):
        hist[i * HIST_NB_BINS: (i + 1) * HIST_NB_BINS] = \
            cv2.calcHist(frame, [i], None, [HIST_NB_BINS], [0, 256])
    hist = cv2.normalize(hist, hist)
    return hist

def get_Histdiff(hist, prev_hist):
    return cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CHISQR)

def is_Hist_diff(img, prev_img):
    hist = get_Hist(img)
    prev_hist = get_Hist(prev_img)
    hist_diff = get_Histdiff(prev_hist, hist)

    is_diff = False
    if hist_diff > HIST_DIFF_THRESHOLD:
        is_diff = True

    return is_diff


def get_frame(n):
    datasets_path = './model/dataset/images/'
    datasets = os.listdir(datasets_path)
    i = n % len(datasets)
    return cv2.imread(datasets_path + datasets[i])


def main():

    # get hard exmaple mining algorithm from config
    hard_example_mining = JointInference.get_hem_algorithm_from_config(
        threshold_img=0.9
    )

    inference_instance = JointInference(
        estimator=Estimator,
        hard_example_mining=hard_example_mining
    )

    fps = 1
    nframe = 0
    img_prev = None
    start_time = time.time() * 1000
    while 1:
        if (time.time() * 1000 - start_time) < (1000 / fps):
            continue
        start_time = time.time() * 1000

        img = get_frame(nframe)
        if img is None:
            LOG.info(f"image not found!")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if nframe == 0:
            is_diff = True
        else:
            is_diff = is_Hist_diff(img_rgb, img_prev)
        img_prev = img_rgb
        if not is_diff:
            continue

        cpu_freq_ratio = open("./freq.txt").readline().strip()
        LOG.info(f"current frame index is {nframe}")
        is_hard_example, final_result, edge_result, cloud_result = (
            inference_instance.inference(img_rgb, float(cpu_freq_ratio))
        )
        LOG.info(cloud_result)
        nframe += 1
        '''
        output_deal(
            final_result,
            is_hard_example,
            cloud_result,
            edge_result,
            nframe,
            img_rgb
        )
        '''


if __name__ == '__main__':
    main()
