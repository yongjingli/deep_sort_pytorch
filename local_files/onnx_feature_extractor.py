import onnxruntime
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os.path as osp
import sys


class OnnxExtractor():
    def __init__(self, onnx_path, batch_size=1):
        self.onnx_path = onnx_path

        self.sess = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name

        self.onnx_file = onnx_path
        self.net_w = 128
        self.net_h = 256
        self.batch_size = batch_size

        self.img_t = None
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.std_inv = 1 / self.std

    def img_pre_process(self, cv_img):
        # resize
        cv_img = cv2.resize(cv_img, (self.net_w, self.net_h))

        if True:
            cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)  # inplace

        # normalize
        cv_img = cv_img.copy().astype(np.float32)
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.std_inv = 1 / np.float64(self.std.reshape(1, -1))

        cv2.subtract(cv_img, self.mean, cv_img)  # inplace
        cv2.multiply(cv_img, self.std_inv, cv_img)  # inplace

        self.img_t = cv_img.transpose(2, 0, 1)  # to C, H, W
        self.img_t = np.ascontiguousarray(self.img_t)
        self.img_t = np.expand_dims(self.img_t, axis=0)
        return self.img_t

    def get_onnx_prediction(self, input_img):
        feats = self.sess.run(None, {self.input_name: input_img})
        return feats

    def infer_cv_img(self, cv_img):
        img_input = self.img_pre_process(cv_img)
        prediction = self.get_onnx_prediction(img_input)[0]

        return prediction

    def infer_batch_cv_imgs(self, cv_imgs):
        feats = []
        for cv_img in cv_imgs:
            feat = self.infer_cv_img(cv_img)
            feats.append(feat)
        feats = np.concatenate(feats, axis=0)
        return feats

    def infer_batch_cv_imgs(self, cv_imgs):
        predictions = []
        for offset in range(0, len(cv_imgs), self.batch_size):
            cur_imgs = cv_imgs[offset:offset + self.batch_size]
            img_inputs = []
            for cur_img in cur_imgs:
                img_input = self.img_pre_process(cur_img)
                img_inputs.append(img_input)

            if len(cur_imgs) != self.batch_size:
                pad_n = self.batch_size - len(cur_imgs)
                pad_inputs = np.ones((pad_n, 3, self.net_h, self.net_w))
                img_inputs.append(pad_inputs)

            img_inputs = np.concatenate(img_inputs, axis=0)
            prediction = self.get_onnx_prediction(img_inputs)[0]
            predictions.append(prediction)

        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions[0:len(cv_imgs), ...]
        return predictions


def compare_cls_feature_distance():
    trt_extractor = OnnxExtractor('/home/liyongjing/Egolee_2021/programs/deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt_sim.onnx')
    img_dir = '/home/liyongjing/Egolee_2021/data/tmp/1'
    img_names = [tmp for tmp in os.listdir(img_dir) if osp.splitext(tmp)[-1] in ['.jpg', 'png']]

    img_names = random.sample(img_names, 100)

    same_id = open('same_id.txt', 'w')
    diff_id = open('diff_id.txt', 'w')

    all_same_id = []
    all_diff_id = []
    for i in tqdm(range(len(img_names)-1)):
        for j in range(i+1, len(img_names), 1):
            img_path_0 = osp.join(img_dir, img_names[i])
            img_path_1 = osp.join(img_dir, img_names[j])

            img_0 = cv2.imread(img_path_0)
            img_1 = cv2.imread(img_path_1)

            a = trt_extractor.infer_cv_img(img_0)
            b = trt_extractor.infer_cv_img(img_1)

            cos = np.dot(a, b.T)[0][0]
            sss = img_names[i]+'---'+img_names[j]+'===='+str(cos)+'\n'
            if img_names[i][:5] == img_names[j][:5]:
                same_id.write(sss)
                all_same_id.append(cos)
            else:
                diff_id.write(sss)
                all_diff_id.append(cos)

    same_id.close()
    diff_id.close()
    print('same_id_average:', sum(all_same_id)/len(all_same_id))
    print('diff_id_average:', sum(all_diff_id) / len(all_diff_id))


if __name__ == '__main__':
    compare_cls_feature_distance()
    exit(1)

    img = cv2.imread('/home/liyongjing/Egolee_2021/data/tmp/1/00031_1_000008.jpg')
    trt_extractor = OnnxExtractor('/home/liyongjing/Egolee_2021/programs/deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt_sim.onnx')
    trt_feats = trt_extractor.infer_cv_img(img)


    sys.path.insert(0, '../')
    from deep_sort.deep.feature_extractor import Extractor
    origin_extractor = Extractor('/home/liyongjing/Egolee_2021/programs/deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt.t7')
    img = img[:, :, (2, 1, 0)]
    origin_feats = origin_extractor([img]) 
