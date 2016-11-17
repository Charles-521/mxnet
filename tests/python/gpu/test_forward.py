import os
import mxnet as mx
import numpy as np
import skimage.io
import skimage.transform
from mxnet.test_utils import *

def GetModel():
    if not os.path.isdir("model/"):
        os.system("mkdir model/")
    if not os.path.exists('model/inception-v3.tar.gz'):
        os.system("wget http://data.mxnet.io/models/imagenet/inception-v3.tar.gz -P model/")
        os.chdir("./model")
        os.system("tar -xf inception-v3.tar.gz --strip-components 1")
        os.chdir("..")

def GetTestData(shape):
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/test_images.tar.gz'):
        os.system("wget http://data.mxnet.io/data/test_images.tar.gz -P data/")
        os.chdir("./data")
        os.system("tar -xf test_images.tar.gz")
        os.chdir("..")
    if not os.path.exists('data/inception-v3-dump.npz'):
        os.system("wget http://data.mxnet.io/data/inception-v3-dump.npz -P data/")
    img_list = []
    for img in sorted(os.listdir('data/test_images/')):
        img = skimage.io.imread('data/test_images/'+img)
        short_egde = min(img.shape[:2])
        yy = int((img.shape[0] - short_egde) / 2)
        xx = int((img.shape[1] - short_egde) / 2)
        img = img[yy : yy + short_egde, xx : xx + short_egde]
        img = skimage.transform.resize(img, shape)
        img_list.append(img)
    return mx.nd.transpose(mx.nd.array(img_list, dtype=np.float32), axes=(0, 3, 1, 2)) - 128

def test_consistency(dump=False):
    GetModel()
    data = GetTestData((299, 299))
    if not dump:
        gt = {n: mx.nd.array(a) for n, a in np.load('data/inception-v3-dump.npz').items()}
    else:
        gt = None
    sym, arg_params, aux_params = mx.model.load_checkpoint('model/Inception-7', 1)
    arg_params['data'] = data
    arg_params['softmax_label'] = np.random.randint(low=1, high=1000, size=(data.shape[0],))
    ctx_list = [{'ctx': mx.gpu(0), 'data': data.shape, 'type_dict': {'data': data.dtype}},
                {'ctx': mx.cpu(0), 'data': data.shape, 'type_dict': {'data': data.dtype}}]
    gt = check_consistency(sym, ctx_list, arg_params=arg_params, aux_params=aux_params,
                           tol=0.01, grad_req='null', raise_on_err=False, ground_truth=gt)
    if dump:
        np.savez('data/inception-v3-dump.npz', **{n: a.asnumpy() for n, a in gt.items()})

if __name__ == '__main__':
    #test_forward_inception()
    test_consistency(False)
