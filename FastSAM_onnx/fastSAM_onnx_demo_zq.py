#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
from random import randint
import random
import cv2
import numpy as np
import onnxruntime as ort
from math import exp

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

meshgrid = []

class_num = 1
head_num = 3
strides = [8, 16, 32]
map_size = [[80, 80], [40, 40], [20, 20]]
nms_thresh = 0.5
object_thresh = 0.25

input_imgH = 640
input_imgW = 640
mask_num = 32

dfl_num = 16

ort_session = ort.InferenceSession('./FastSAM_S.onnx')


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, mask):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.mask = mask


def GenerateMeshgrid():
    for index in range(head_num):
        for i in range(map_size[index][0]):
            for j in range(map_size[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nms_thresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []
    output = []
    for i in range(len(out)):
        print(out[i].shape)
        output.append(out[i].reshape((-1)))

    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW

    gridIndex = -2
    cls_index = 0
    cls_max = 0

    for index in range(head_num):
        reg = output[index * 2 + 0]
        cls = output[index * 2 + 1]
        msk = output[head_num * 2 + index]

        for h in range(map_size[index][0]):
            for w in range(map_size[index][1]):
                gridIndex += 2

                if 1 == class_num:
                    cls_max = sigmoid(cls[0 * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])
                    cls_index = 0
                else:
                    for cl in range(class_num):
                        cls_val = cls[cl * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w]
                        if 0 == cl:
                            cls_max = cls_val
                            cls_index = cl
                        else:
                            if cls_val > cls_max:
                                cls_max = cls_val
                                cls_index = cl
                    cls_max = sigmoid(cls_max)

                if cls_max > object_thresh:
                    regdfl = []
                    for lc in range(4):
                        sfsum = 0
                        locval = 0
                        for df in range(dfl_num):
                            temp = exp(reg[((lc * dfl_num) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])
                            reg[((lc * dfl_num) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w] = temp
                            sfsum += temp

                        for df in range(dfl_num):
                            sfval = reg[((lc * dfl_num) + df) * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w] / sfsum
                            locval += sfval * df
                        regdfl.append(locval)

                    x1 = (meshgrid[gridIndex + 0] - regdfl[0]) * strides[index]
                    y1 = (meshgrid[gridIndex + 1] - regdfl[1]) * strides[index]
                    x2 = (meshgrid[gridIndex + 0] + regdfl[2]) * strides[index]
                    y2 = (meshgrid[gridIndex + 1] + regdfl[3]) * strides[index]

                    xmin = x1 * scale_w
                    ymin = y1 * scale_h
                    xmax = x2 * scale_w
                    ymax = y2 * scale_h

                    xmin = xmin if xmin > 0 else 0
                    ymin = ymin if ymin > 0 else 0
                    xmax = xmax if xmax < img_w else img_w
                    ymax = ymax if ymax < img_h else img_h

                    mask = []
                    for m in range(mask_num):
                        mask.append(msk[m * map_size[index][0] * map_size[index][1] + h * map_size[index][1] + w])

                    box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax, mask)
                    detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def seg_postprocess(out, predbox, img_h, img_w):
    print('seg_postprocess ... ')
    protos = np.array(out[9][0])

    c, mh, mw = protos.shape
    seg_mask = np.zeros(shape=(mh, mw, 3))
    mask_contour = []

    for i in range(len(predbox)):
        masks_in = np.array(predbox[i].mask).reshape(-1, c)
        masks = (masks_in @ protos.reshape(c, -1))

        masks = 1 / (1 + np.exp(-masks))
        masks = masks.reshape(mh, mw)

        xmin = int(predbox[i].xmin / img_w * mw + 0.5)
        ymin = int(predbox[i].ymin / img_h * mh + 0.5)
        xmax = int(predbox[i].xmax / img_w * mw + 0.5)
        ymax = int(predbox[i].ymax / img_h * mh + 0.5)
        classId = predbox[i].classId
        mask_color = random_color()
        gray_mask = np.zeros(shape=(mh, mw, 1), dtype='uint8')
        for h in range(ymin, ymax):
            for w in range(xmin, xmax):
                if masks[h, w] > 0.5:
                    seg_mask[h, w, :] = mask_color
                    gray_mask[h, w] = 255

        gray_mask = cv2.resize(gray_mask, (img_w, img_h))
        ret, binary = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_contour.append(contours)

    seg_mask = cv2.resize(seg_mask, (img_w, img_h))
    seg_mask = seg_mask.astype("uint8")
    return seg_mask, mask_contour


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255.0

    return image


def detect(img_path):

    origin_image = cv2.imread(img_path)
    img_h, img_w = origin_image.shape[:2]
    image = precess_image(origin_image, input_imgW, input_imgH)

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    pred_results = (ort_session.run(None, {'data': image}))

    out = []
    for i in range(len(pred_results)):
        out.append(pred_results[i])
    predbox = postprocess(out, img_h, img_w)

    mask, mask_contour = seg_postprocess(out, predbox, img_h, img_w)

    print('obj num is :', len(predbox))
    show_rect = True
    result_img = cv2.addWeighted(origin_image, 0.8, mask, 0.2, 0.0)
    for i in range(len(predbox)):
        if show_rect:
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            classId = predbox[i].classId
            score = predbox[i].score

            # cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), random_color(), 2)
            ptext = (xmin, ymin + 10)
            title = "%.2f" % score
            cv2.putText(result_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawContours(result_img, mask_contour[i], -1, random_color(), 2)

    cv2.imwrite(r'./test_onnx_result.jpg', result_img)


if __name__ == '__main__':
    print('This is main ....')
    GenerateMeshgrid()
    img_path = './test.jpg'
    detect(img_path)