import tensorrt as trt   # tensorRT 7.0
import pycuda.driver as cuda
import pycuda.autoinit

from datetime import datetime
import numpy as np
import cv2
import time
import os
from tqdm import tqdm
import random
import os.path as osp
import sys

# You can set the logger severity higher to suppress messages (or lower to display more messages).
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # set batch size


def build_engine_trt(model_file, max_batch_size=1):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
            network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        builder.max_batch_size = max_batch_size
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def GiB(val):
    return val * 1 << 30


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        bindig_shape = tuple(engine.get_binding_shape(binding))
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(bindig_shape, dtype)
        # print('\tAllocate host buffer: host_mem -> {}, {}'.format(host_mem, host_mem.nbytes))  # host mem

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # print('\tAllocate device buffer: device_mem -> {}, {}'.format(device_mem, int(device_mem))) # device mem

        # print('\t# Append the device buffer to device bindings.......')
        bindings.append(int(device_mem))
        # print('\tbindings: ', bindings)

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            # print('____HostDeviceMem(host_mem, device_mem)): {}, {}'.format(HostDeviceMem(host_mem, device_mem),type(HostDeviceMem(host_mem, device_mem))))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # print("This is the output!")
            outputs.append(HostDeviceMem(host_mem, device_mem))
        # print("----------------------end allocating one binding in the onnx model-------------------------")

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TrtExtractor(object):
    def __init__(self, onnx_path, batch_size=2):
        print(' TrtExtractor Initial Start...')
        self.onnx_file = onnx_path
        self.net_w = 64
        self.net_h = 128
        # self.net_w = 128
        # self.net_h = 256
        self.batch_size = batch_size

        self.img_t = None
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.std_inv = 1 / self.std

        self.engine = build_engine_trt(onnx_path, batch_size)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        print(' TrtExtractor Initial Done...')

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

    def infer_cv_img(self, cv_img):
        img_input = self.img_pre_process(cv_img)
        prediction = self.get_trt_prediction(img_input)[0]

        return prediction

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
            prediction = self.get_trt_prediction(img_inputs)[0]
            predictions.append(prediction)

        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions[0:len(cv_imgs), ...]
        return predictions

    def get_trt_prediction(self, input_img):
        np.copyto(self.inputs[0].host, input_img)
        feats = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=self.batch_size)
        # feats.sort(key=lambda x: x.shape[2], reverse=True)
        # do something  or return copy feature
        feats = [feat.copy() for feat in feats]
        return feats


def compare_cls_feature_distance():
    trt_extractor = TrtExtractor('/home/liyongjing/Egolee_2021/programs/FastMOT-master/fastmot/models/osnet_x0_25_msmt17.onnx', 16)
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

            a = trt_extractor.infer_batch_cv_imgs([img_0])
            b = trt_extractor.infer_batch_cv_imgs([img_1])

            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
            # feats = trt_extractor.infer_batch_cv_imgs([img_0, img_1])
            # print(feats.shape)
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
    # compare_cls_feature_distance()

    # exit(1)
    img = cv2.imread('/home/liyongjing/Egolee_2021/data/tmp/1/00031_1_000008.jpg')
    img2 = cv2.imread('/home/liyongjing/Egolee_2021/data/tmp/1/00033_1_000512.jpg')
    img3 = cv2.imread('/home/liyongjing/Egolee_2021/data/tmp/1/00029_1_000504.jpg')

    batch_size = 2

    trt_extractor = TrtExtractor('/home/liyongjing/Egolee_2021/programs/deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt_sim.onnx', batch_size)
    # trt_feats = trt_extractor.infer_cv_img(img)
    # imgs = [img, img2, img3]
    imgs = [img, img2, img3]
    trt_feats = trt_extractor.infer_batch_cv_imgs(imgs)
    # print(trt_feats.shape)
    # exit(1)
    for i in range(len(imgs)):
        print('output {} :'.format(i))
        trt_feat = trt_feats[i, ...][:10]
        print(trt_feat)
    print('Process Done.')

    # sys.path.insert(0, '../')
    # from deep_sort.deep.feature_extractor import Extractor
    # origin_extractor = Extractor('/home/liyongjing/Egolee_2021/programs/deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt.t7')
    # img = img[:, :, (2, 1, 0)]
    # origin_feats = origin_extractor([img]) 
