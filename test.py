import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import CANDI
from utils import *
import pandas as pd
from scipy import io
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="CANDI_Test")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--output_size", type=int, default=10, help='The output size of average pooling in a channel attention module')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = CANDI(channels=1, k=opt.output_size)
    
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_%d.pth'% (opt.test_noiseL))))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.*'))
    files_source.sort()
    # process data
    psnr_test = 0
    ssim_test = 0
    for ff in files_source:
        # image
        Img = cv2.imread(ff,cv2.IMREAD_GRAYSCALE)
        Img = normalize(np.float32(Img[:,:]))

        Img = np.expand_dims(np.expand_dims(Img, 0), 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)

        Out = (Out*255+0.5).type(torch.uint8)
        Out = Out / 255.
        psnr = batch_PSNR(Out, ISource, 1.)
        ssim = batch_SSIM(Out, ISource, 1.)
        psnr_test += psnr
        ssim_test += ssim
        print("%s PSNR %f SSIM %f" % (ff, psnr, ssim))
        save_img = (Out*255).cpu().numpy().astype(np.uint8) 
        cv2.imwrite(os.path.join('result', opt.test_data, 'result_'+ff.split('/')[-1]), np.squeeze(save_img))
        
        INoisy = torch.clamp(INoisy, 0., 1.)
        noisy_img = (INoisy*255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join('result', opt.test_data, 'noisy_'+ff.split('/')[-1]), np.squeeze(noisy_img))
       

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print("\nSSIM on test data %f" % ssim_test)

if __name__ == "__main__":
    main()
