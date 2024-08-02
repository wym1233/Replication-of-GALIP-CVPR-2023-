import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
ROOT_PATH = '/data/wym123/24RSP_GALIP/selfckptdir'
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from lib.utils import load_models,save_models_opt,save_models,load_npz,params_count
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import sample_one_batch as sample, test as test, train as train
from lib.datasets import get_fix_data
import clip as clip

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='/code/GALIP-main/code/cfg/birds.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='new',
                        help='file path to log directory')
    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=480,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args):
    # prepare dataloader, models, data
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    print('**************G_paras: ',params_count(netG))
    path = osp.join('/data/wym123/24RSP_GALIP/selfckptdir/saved_models/bird/GALIP_nf64_gpuMP_True_bird_256_2024_07_30_08_43_46','state_epoch_600.pth')
    netG, netD, netC = load_models(netG, netD, netC, path)

    noise = torch.randn(1, 100).to(args.device)

    # fixed_sent
    # caption="A young girl with colorful face painting and geometric sweater."
    # caption="The image features a close-up of a young girl with vibrant facial paint and colorful clothing, showcasing childlike innocence and creativity"
    # caption="This image features a close-up of a young girl with vibrant facial paint and colorful clothing, showcasing childlike innocence and creativity. The contrast between her natural skin tone and the bright colors on her face and sweater creates a captivating visual experience"
    # caption="The image depicts a half-body portrait of a young girl. Her eyes are large and expressive, gazing directly at the camera with a natural expression and a slight smile on her lips.The girl has dark skin with delicate features. Her skin appears smooth, although there are faint dark circles under her eyes.Her hair is black and tied into a low ponytail, secured with a colorful striped hairband.Multicolored face paint adorns her cheeks and forehead. The designs include various shapes such as flowers and stars in colors like yellow, red, blue, and green. A larger circular design stands out in the center, surrounded by smaller patterns.She wears a red sweater decorated with green geometric shapes. A colorful bead necklace also hangs around her neck.Overall, this vibrant and lively photo captures the innocent smile and unique facial artwork of the little girl, giving off a warm and joyful vibe."
    
    # caplist=["a dog.",
    # "A dog sits on the grass.",
    # 'A dog is smiling',
    # 'A dog plays with a ball',
    # 'An empty room',
    # 'A crowded room',
    # 'A cosy room',
    # 'A party is taking place in the room',
    # 'Three people stood together and discussed the problem',
    # 'A woman stands in front of the fireplace']

    image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            ])

    norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def get_imgs(img_path, bbox=None, transform=None, normalize=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])
        if transform is not None:
            img = transform(img)
        if normalize is not None:
            img = normalize(img)
        return img

    caplist=['this is a small bird with a black face and black lower body but his upper head and chest is yellow']
    i=599

    imgpth='/data/wym123/24RSP_GALIP/birds/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0012_8443.jpg'
    img=get_imgs(imgpth,transform=image_transform, normalize=norm).unsqueeze(dim=0).to(args.device)

   
   


   
    # for caption in caplist:
    #     i=i+1
    #     tokens = clip.tokenize(caption,truncate=True).to(args.device)
    #     sent_emb,words_embs = text_encoder(tokens)

    #     with torch.no_grad():
    #         fake = netG(img,noise, sent_emb, eval=True)
    #     img_save_path = "/code/GALIP-main/code/genimgs/"+str(i)+".png"
    #     vutils.save_image(fake.data, img_save_path, value_range=(-1, 1), normalize=True)


    



if __name__ == "__main__":
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

            # torch.distributed.init_process_group(backend="nccl")
            
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

