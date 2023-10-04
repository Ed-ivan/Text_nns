'''
将 text , 或者 img 先 经过clip的 encoder
这一部分 ， 就先不用那个pl库了  ， 因为需要使用.fit()
目前还差一个保存 img_name 的
'''

import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import lightning.pytorch as pl
from tqdm import tqdm
from Utils.utils import save_embeddings
import os

class ClipEncoder(pl.LightningModule):
    def __init__(self,device,save_dir,mode='image'):
        self.mode=mode
        self.model,self.clip_preprocess=clip.load("ViT-B/32", device=device)
        # 还有就是需要进行预处理  ，这个地方感觉
        self.preprocess=transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              self.clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              self.clip_preprocess.transforms[4:])
        #
        self.save_dir=save_dir



    def training_step(self,batch,batch_idx):
        #traing_step define the loop    这个暂时不用
        image_features=self.model.encode_image(batch)
        #text_features=self.model.encode_text()

    def image_preprocess(self,mode,batch):
        assert  mode is "image" , "this is for image preprocess"+f''+self.save_dir
        image_features=self.preprocess(batch)
        return image_features


    def get_embeddings(self,loader):
        for i , batch in enumerate(tqdm(loader,desc='Building the image clip_embeddings',total=len(loader))):
            if (self.mode=='text'):
                batch_size=batch.size(0)

                embeddings=self.model.encode_text(batch)
                for j in range(batch_size):
                    filename = f'img_embeddings/-img{i * batch_size+j:05d}.p'
                    filepath = os.path.join(self.save_dir, filename)
                    # 这个地方感觉还是需要修改一下  ， 不能只是简单的保存embeddings ， 还需要保存 image的 名称 ， 因为在 source_root中
                    #
                    save_embeddings(filepath,embeddings[j])
            else:
                batch_size=batch[0]
                img, img_name = batch
                assert self.mode is "image", "this is for image preprocess" + f'' + self.save_dir
                image_features = self.preprocess(img)
                embeddings=self.model.encode_image(image_features)
                for j in range(batch_size):
                    filename = f'img_embeddings/-img{i * batch_size+j:05d}.p'
                    filepath = os.path.join(self.save_dir, filename)
                    # 这个地方感觉还是需要修改一下  ， 不能只是简单的保存embeddings ， 还需要保存 image的 名称 ， 因为在 source_root中
                    #
                    save_embeddings(filepath,embeddings[j])

                # 应该是分开存储这些 embeddings




# model, preprocess = clip.load("ViT-B/32", device=device)
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]