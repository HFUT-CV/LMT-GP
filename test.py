import os
import torch
import torch.nn as nn
import torchvision.transforms
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from adamp import AdamP
# my import
from model_mimo import SemiLL
from dataset_all import TestData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
bz = 1
model_root = '/run/media/chinn/新加卷/GP实验记录/ECCV_version/data/best.pth'
input_root = '/home/chinn/Download/data/DICM/low/'
save_path = 'result/DICM'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)['state_dict']
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=bz)

model = SemiLL()

model = nn.DataParallel(model, device_ids=[0])
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint)
# optimizer.load_state_dict(checkpoint['optimizer_dict'])
# epoch = checkpoint['epoch']
model.eval()
import time
print('START!')
if 1:
    print('Load model successfully!')
    for data_idx, data_ in enumerate(data_load):
        data_input,  hs, ws = data_

        data_input = Variable(data_input)

        print(data_idx)
        with torch.no_grad():
            start_time = time.perf_counter()
            results, _ = model(data_input)
            end__time = time.perf_counter()
            print(end__time-start_time)
            result = results[0]
            result = torch.clamp(result, 0, 1)

            result = result[:, :, :hs[0], :ws[0]]
            name = Mydata_.A_paths[data_idx].split('/')[-1]
            print(name)

            # temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res = result[0]
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = torchvision.transforms.ToPILImage()(temp_res)
            # temp_res = (temp_res*255).astype(np.uint8)
            # temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
            print('result saved!')

print('finished!')
