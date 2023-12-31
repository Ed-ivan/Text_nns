import os.path
from torch.utils.data import Dataset
from PIL import Image
from Utils.ldm_utils import instantiate_from_config
from Utils.utils import  make_dataset
from ..transform import  img_transform
from utils import data_utils
import numpy as np
import torch

# 对应 23种设计模式中的指派

class SHHQDataset(Dataset):
    def __init__(self,source_root):
        self.source_paths=make_dataset(source_root)


    def __len__(self):
        return len(self.source_paths)
    def  _get_path(self):
        return self.source_paths
    def __getitem__(self, index):
        img_path = self.source_paths[index]
        img = Image.open(img_path)
        img = img.convert('RGB')

        # to_path = self.target_paths[index]
        # to_im = Image.open(to_path).convert('RGB')
        # if self.target_transform:
        # to_im = self.target_transform(to_im)
        # else:
        # from_im = to_im
        # 这里面需要同时返回image的  名称
        img_name=os.path.split(img_path)
        return img,img_name
class SHHQ_Train(SHHQDataset):
    def __init__(self,source_root,flag,transform=None):
        super(SHHQ_Train, self).__init__(source_root)
        if  transform is not None:
            self.transform=instantiate_from_config(transform)
        self.flag=flag
    def __getitem__(self, index):
        im,img_name=super().__getitem__(index)
        assert  self.flag is  'train','something goes wrong!'
        im=self.transform.get_transforms[self.flag](im)
        return im,img_name

class SHHQ_Val(SHHQDataset):
    # not  implement well
    def __init__(self, source_root):
        super(SHHQ_Train, self).__init__(source_root)

    '''
    latent 中存储的是embeddings , img_name
    所以这个地方还需要 修改一下 ， 加上 img_name 但是应该怎么构造呢 ？
    (需要通过这个构建 data_pool), 先假设
     for batch in (pbar := tqdm(loader, desc='loading feature embeddings',
                                       total=total)):
                if 'patch' in batch:
                    patches = batch['patch']
    '''
class LatentsDataset(Dataset):

    def __init__(self, latents_path):
        self.latent_path=latents_path
        self.latents=torch.load(self.latent_path)

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        latents=self.latents[index]
        embeddings=latents['embeddings']
        ids=latents['content_ids']
        # 有了  这个id就可以找到 对应的latent 对应的原始内容
        return embeddings,ids



