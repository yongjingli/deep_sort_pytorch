import os
import cv2
import time
import argparse
# import torch
import warnings
import numpy as np

import sys
sys.path.insert(0, '../')

# from detector import build_detector
# from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

sys.path.insert(0, '/home/liyongjing/Egolee_2021/programs/yolov5-master_4/local_files')
from yolo_trt_infer import YoloDetectionTrt
from yolo_onnx_infer import YoloDetectionOnnx
from deep_sort_self import DeepSortSelf


class TrackPersonWtihFace():
    def __init__(self, person_box, track_id, has_face, face_box=None, face_iou=0, face_up_dist=0, face_w_ratio=0):
        self.person_box = person_box
        self.track_id = track_id

        self.has_face = has_face
        self.face_box = face_box
        self.face_iou = face_iou
        self.face_up_dist = face_up_dist
        self.face_w_ratio = face_w_ratio


def get_person_face_box(person_bboxes_xyxy, track_ids, face_bboxes_xyxy):
    person_face_boxes = []
    face_boxes_not_track = []
    face_ids = []
    person_face_boxes_nms = []
    for i, (person_bbox, track_id) in enumerate(zip(person_bboxes_xyxy, track_ids)):
        p_x, p_y, p_x2, p_y2 = person_bbox
        p_w = p_x2 - p_x
        p_h = p_y2 - p_y

        max_iou = 0
        max_up_dist = 0
        max_w_ration = 0
        face_id = -1
        person_face_box = None

        for j, face_box in enumerate(face_bboxes_xyxy):
            f_x, f_y, f_x2, f_y2 = face_box
            f_w = f_x2 - f_x
            f_h = f_y2 - f_y
            face_area = f_w * f_h

            xx1 = max(f_x, p_x)
            yy1 = max(f_y, p_y)
            xx2 = min(f_x2, p_x2)
            yy2 = min(f_y2, p_y2)
            iou_w = max(xx2 - xx1, 0)
            iou_h = max(yy2 - yy1, 0)

            up_dist = f_h/abs(f_y - p_y)
            iou = iou_h * iou_w / face_area  # 0-1
            w_ration = min(f_w, p_w) / max(f_w, p_w)

            if iou > 0.9:
                iou = 1.0

            if iou < 0.3:
                iou = -1

            if up_dist < 0.4:
                iou = -1

            if iou > max_iou:
                max_iou = iou
                max_up_dist = up_dist
                max_w_ration = w_ration
                person_face_box = face_box
                face_id = j
            elif iou == max_iou:
                if up_dist > max_up_dist:
                    max_iou = iou
                    max_up_dist = up_dist
                    max_w_ration = w_ration
                    person_face_box = face_box
                    face_id = j
                elif up_dist == max_up_dist:
                    if max_w_ration > w_ration:
                        max_iou = iou
                        max_up_dist = up_dist
                        max_w_ration = w_ration
                        person_face_box = face_box
                        face_id = j

        if face_id != -1:
            person_face_box = TrackPersonWtihFace(person_bbox, track_id, True, person_face_box, max_iou, max_up_dist, max_w_ration)
            person_face_boxes.append(person_face_box)
            face_ids.append(face_id)
        else:
            person_face_box = TrackPersonWtihFace(person_bbox, track_id, False, None, 0, 0, 0)
            person_face_boxes_nms.append(person_face_box)

    # person face repeat filter
    for i, face_box in enumerate(face_bboxes_xyxy):
        if i not in face_ids:
            face_boxes_not_track.append(face_box)

        repeat_person_face_boxes = []
        for j, face_id in enumerate(face_ids):
            if face_id == i:
                repeat_person_face_boxes.append(person_face_boxes[j])

        repeat_person_face_boxes = sorted(repeat_person_face_boxes, key=lambda x: (x.face_iou, x.face_up_dist, x.face_w_ratio), reverse=True)
        for k, repeat_person_face_box in enumerate(repeat_person_face_boxes):
            if k != 0:
                repeat_person_face_box.has_face = False
            person_face_boxes_nms.append(repeat_person_face_box)
    return person_face_boxes_nms


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_person_face_boxes(img, person_face_boxes):
    for i, person_face_box in enumerate(person_face_boxes):
        person_box = person_face_box.person_box
        track_id = person_face_box.track_id
        if person_face_box.has_face:
            face_box = person_face_box.face_box
        else:
            face_box = None

        x1, y1, x2, y2 = [int(i) for i in person_box]
        id = int(track_id)
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        if face_box is not None:
            x1, y1, x2, y2 = [int(i) for i in face_box]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        # use_cuda = args.use_cuda and torch.cuda.is_available()
        # if not use_cuda:
        #     warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        # self.detector = build_detector(cfg, use_cuda=use_cuda)

        self.detector = YoloDetectionOnnx(cfg.det_model_cfg['onnx_path'], cfg.det_model_cfg['batch_size'],
                                             anchor=cfg.det_model_cfg['anchors'], stride=cfg.det_model_cfg['strides'])

        from trt_feature_extractor import TrtExtractor
        from onnx_feature_extractor import OnnxExtractor
        extractor = OnnxExtractor(cfg.deep_sort_cfg['onnx_path'], cfg.deep_sort_cfg['batch_size'])
        self.deepsort = DeepSortSelf(extractor,
                                    max_dist=cfg.deep_sort_cfg['max_dist'], min_confidence=cfg.deep_sort_cfg['min_confidence'],
                                    nms_max_overlap=cfg.deep_sort_cfg['nms_max_overlap'], max_iou_distance=cfg.deep_sort_cfg['max_iou_distance'],
                                    max_age=cfg.deep_sort_cfg['max_age'], n_init=cfg.deep_sort_cfg['n_init'], nn_budget=cfg.deep_sort_cfg['nn_budget'])

        # self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        # self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            # bbox_xywh, cls_conf, cls_ids = self.detector(im)
            det_result = self.detector.infer_cv_img(ori_im)
            # print(det_result.shape)
            # exit(1)
            if len(det_result) <= 0:
                det_result = np.empty((0, 6))

            cls_conf = det_result[:, 4]
            cls_ids = det_result[:, 5]

            # face boxes
            mask = cls_ids == 1
            face_bbox_xyxy = det_result[:, 0:4][mask]

            det_result[:, 2:4] = det_result[:, 2:4] - det_result[:, 0:2]  # xyxy2xywh
            det_result[:, 0:2] = det_result[:, 0:2] + det_result[:, 2:4] * 0.5
            # print(det_result.shape)
            bbox_xywh = det_result[:, 0:4]

            end_yolo_time = time.time() - start
            self.logger.info("yolo det time:{:.03f}s".format(end_yolo_time))

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                # ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                # #
                person_face_boxes = get_person_face_box(bbox_xyxy, identities, face_bbox_xyxy)
                ori_im = draw_person_face_boxes(ori_im, person_face_boxes)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(0)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default='')
    parser.add_argument("--config_detection", type=str, default="../configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="../configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    onnx_path = '/home/liyongjing/Egolee_2021/programs/yolov5-master_4/runs/train/train_person_face_0624/weights/best.onnx'
    anchors = np.array(
        [[10., 13., 16., 30., 33., 23.],
         [30., 61., 62., 45., 59., 119.],
         [116., 90., 156., 198., 373., 326.]], dtype=np.float32)

    det_model_cfg = {
                'onnx_path': onnx_path,
                'anchors': anchors,
                'strides': [8., 16., 32.],
                'batch_size': 1
    }

    deep_sort_cfg = {
        'onnx_path': '/home/liyongjing/Egolee_2021/programs/deep-person-reid-master/local_files/osnet_ain_x1_0.onnx',
        'batch_size': 1,
        'max_dist': 0.2,
        'min_confidence': 0.3,
        'nms_max_overlap': 0.5,
        'max_iou_distance': 0.7,
        'max_age': 20,
        'n_init': 3,
        'nn_budget': 10
        }

    cfg.det_model_cfg = det_model_cfg
    cfg.deep_sort_cfg = deep_sort_cfg

    args.VIDEO_PATH = '/home/liyongjing/Egolee_2021/data/src_person_car/person_car.mp4'
    args.display = True
    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
      
