# import os


# import pyiqa
# import torch

# # list all available metrics
# # print(pyiqa.list_models())

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # create metric with default setting
# # iqa_metric = pyiqa.create_metric('topiq_nr', device=device,pretrained=True,pretrained_model_path="/data/wym123/24RSP_GALIP/pretrainckpts/cfanet_nr_koniq_res50-9a73138b.pth")
# # iqa_metric = pyiqa.create_metric('topiq_fr', device=device,pretrained=True,pretrained_model_path="/data/wym123/24RSP_GALIP/pretrainckpts/cfanet_fr_kadid_res50-2c4cc61d.pth")
# # check if lower better or higher better
# iqa_metric = pyiqa.create_metric('ms_ssim', device=device)
# print(iqa_metric.lower_better)

# # example for iqa score inference
# # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
# # score_fr = iqa_metric(img_tensor_x, img_tensor_y)

# # img path as inputs.
# score_fr = iqa_metric('/code/GALIP-main/code/genimgs/demo2.png','/code/GALIP-main/code/genimgs/demo3.png')

# print(score_fr)


import os, sys
import os.path as osp
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p, get_rank
from lib.datasets import prepare_data

from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist

from lib.utils import load_netG,load_npz,save_models
from lib.perpare import prepare_dataloaders
from lib.perpare import prepare_models
from lib.modules import test as test
import pyiqa
from pytorch_msssim import ms_ssim
def test(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    print(calculate_FID_CLIP_sim(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size))

def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    
    topiq_nr_metric = pyiqa.create_metric('topiq_nr', device=device,pretrained=True,pretrained_model_path="/data/wym123/24RSP_GALIP/pretrainckpts/cfanet_nr_koniq_res50-9a73138b.pth")
    topiq_fr_metric = pyiqa.create_metric('topiq_fr', device=device,pretrained=True,pretrained_model_path="/data/wym123/24RSP_GALIP/pretrainckpts/cfanet_fr_kadid_res50-2c4cc61d.pth")
    lpips_metric = pyiqa.create_metric('lpips', device=device,pretrained=True,pretrained_model_path="/data/wym123/24RSP_GALIP/pretrainckpts/LPIPS_v0.1_alex-df73285e.pth")
    # msssim_metric = pyiqa.create_metric('ms_ssim', device=device,test_y_channel=False)
    
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])

    norm1 = transforms.Compose([
    transforms.Resize((224, 224)),
    ])

    norm2 = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        ])




    dl_length = dataloader.__len__()
    imgs_num = dl_length * batch_size * times
    pred_arr = np.empty((imgs_num, dims))

    clip_cos = torch.FloatTensor([0.0]).to(device)
    topiq_nr_loss=torch.FloatTensor([0.0]).to(device)
    topiq_fr_loss=torch.FloatTensor([0.0]).to(device)
    msssim_loss=torch.FloatTensor([0.0]).to(device)
    lpips_loss=torch.FloatTensor([0.0]).to(device)

    loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size  + time * dl_length * batch_size
            end = start + batch_size
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            imgs=norm1(imgs)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb,eval=True).float()
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)

                #clipsim
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim

                #fid
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred_arr[start:end] = pred.squeeze().cpu().data.numpy()

                #others
                fake_imgs=norm2(fake_imgs)
                topiq_nr_loss+=torch.mean(topiq_nr_metric(fake_imgs))
                topiq_fr_loss+=torch.mean(topiq_fr_metric(fake_imgs,imgs))

                ssimvalue=torch.mean(ms_ssim( fake_imgs, imgs, data_range=1, size_average=False ))
                if torch.isnan(ssimvalue)==torch.Tensor([True]).to(device):
                    print('Nandata!')
                    print(msssim_metric(fake_imgs,imgs))
                else:
                    msssim_loss+=ssimvalue

                lpips_loss+=torch.mean(lpips_metric(fake_imgs,imgs))


            # update loop information
            loop.update(1)
            if epoch==-1:
                loop.set_description('Evaluating]')
            else:
                loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    
    loop.close()
    # CLIP-score
    clip_score = clip_cos.item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    # topiq_nr,topiq_fr,msssim,lpips
    topiq_nr=topiq_nr_loss.item()/(dl_length*times)
    topiq_fr=topiq_fr_loss.item()/(dl_length*times)
    msssim=msssim_loss.item()/(dl_length*times)
    lpips=lpips_loss.item()/(dl_length*times)

    return fid_value,clip_score,topiq_nr,topiq_fr,msssim,lpips


def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='/code/GALIP-main/code/cfg/coco.yml')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='/data/wym123/24RSP_GALIP/selfckptdir/saved_models/coco/GALIP_nf64_gpuMP_True_coco_256_2024_07_23_01_49_44/state_epoch_450.pth',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='new',
                        help='file path to log directory')
    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=100,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args): 
  
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    state_path = args.pretrained_model_path
    m1, s1 = load_npz(args.npz_path)
    netG = load_netG(netG, state_path, args.multi_gpus, args.train)

    netG.eval()
    test(valid_dl, text_encoder, netG, CLIP4evl, args.device, m1, s1, -1, -1, \
                    args.sample_times, args.z_dim, args.batch_size)


if __name__ == "__main__":
    from lib.utils import merge_args_yaml
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            # torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)



