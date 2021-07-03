import os
import os.path as osp
import sys
import onnx
sys.path.insert(0, '../')
from deep_sort.deep.model import Net
import torch
import numpy as np


def export_net_onnx():
    model_path = '/home/liyongjing/Egolee_2021/programs/deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt.t7'
    model = Net(reid=True)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
    model.load_state_dict(state_dict)

    img_size = (64, 128)
    batch_size = 2
    img = torch.randn(batch_size, 3, *img_size[::-1])
    # img = torch.ones(batch_size, 3, *img_size[::-1])
    model.eval()
    y = model(img)  # dry run

    try:
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = osp.splitext(model_path)[0] + '.onnx'
        torch.onnx.export(model, img, f, verbose=True, opset_version=9, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('====ONNX export success, saved as %s' % f)

        # simpily onnx
        from onnxsim import simplify
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"

        f2 = f.replace('.onnx', '_sim.onnx')  # filename
        onnx.save(model_simp, f2)
        print('====ONNX SIM export success, saved as %s' % f2)

        # check output different between pytorch and onnx: y, y_onnx
        import onnxruntime as rt
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        # net_feed_input = input_all
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(f2)
        y_onnx = sess.run(None, {net_feed_input[0]: img.detach().numpy()})[0]

        # for i, (_y, _y_onnx) in enumerate(zip(y, y_onnx)):
        y_numpy = y.detach().numpy()
        # all_close = np.allclose(_y_numpy, _y_onnx, rtol=1e-05, atol=1e-06)

        # for x, y in zip(y_numpy[0, 0:20], y_onnx[0, 0:20]):
        #     print(x)
        #     print(y)
        #     print('*' * 10)
        #
        print(y_numpy.shape)
        print(y_onnx.shape)

        print(y_numpy[0, 0:20])

        diff = y_numpy - y_onnx

        print('max diff {}'.format(np.max(diff)))
            # assert(np.max(diff) > 1e-5)

        from onnx import shape_inference
        f3 = f2.replace('.onnx', '_shape.onnx')  # filename
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f2)), f3)
        print('====ONNX shape inference export success, saved as %s' % f3)

    except Exception as e:
        print('ONNX export failure: %s' % e)


if __name__ == '__main__':
    export_net_onnx()
