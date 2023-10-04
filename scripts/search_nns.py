'''
ivan
根据给定的text, 找到与之相关的  k个img_embeddings
config : 有 image_embeddings  的路径  ,有text_embeddings 路径
save_dir :保存的路径
(查看)find_nns .yaml文件
'''
import os
import pickle
import sys
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm


def search_nns(dataset_builder, qloader, device='cuda', mode='img', save=False, npatches_perside=None, base_savedir=None, nn_paths=None, corrupts=None, start_id=0, max_its=None):
    assert dataset_builder.searcher is not None
    dset_batch_size = qloader.batch_size

    if save:
        assert base_savedir is not None
        assert npatches_perside is not None
        assert os.path.isdir(os.path.join(base_savedir,'embeddings'))
        if nn_paths is None:
            nn_paths = {}
        if corrupts is None:
            corrupts = []
    ns = 0
    return_ids = {}

    for i, batch in enumerate(tqdm(qloader, desc='Searching nns and saving embeddings', total=len(qloader) if max_its is None else max_its)):
        if max_its is not None and i >= max_its:
            break
        query = batch['patches'].to(device) if mode == 'img' else batch['caption']
        # 不行 ， 看来这块还得改改 ， 就是加载数据集时候
        if isinstance(query, torch.Tensor):
            b, n, *_ = query.shape
            query = rearrange(query, 'b n h w c -> (b n) h w c')
        else:
            b, n = len(query), 1
        caption_sim = mode == 'text'
        results = dataset_builder.search_k_nearest(query, visualize=False, is_caption=caption_sim)
        #等下  ， 感觉有点不对啊，

        if save:
            results = {
                key: results[key].reshape(b, n,*results[key].shape[1:])
                     if isinstance(results[key], np.ndarray) else
                     results[key]
                for key in results
            }

            for j in range(len(results['embeddings'])):
                filename = f'embeddings/{dataset_builder.k}_nns-img{start_id + i * dset_batch_size + j:09d}.p'
                filepath = os.path.join(base_savedir, filename)
                save_it = {npatches_perside: {
                    'embeddings': results['embeddings'][j],
                    'img_ids': results['img_ids'][j],
                    'patch_coords': results['patch_coords'][j],
                    'nn_ids': results['nns'][j]
                }}
                corrupts = save_pkl(filepath=filepath,
                                    save_it=save_it,
                                    npatches_perside=npatches_perside,
                                    corrupts=corrupts,
                                    i=i,j=j,
                                    start_id=start_id,
                                    dset_batch_size=dset_batch_size)

                nn_paths.update({start_id + i * dset_batch_size + j: filename})
                ns += 1
        else:
            ids, counts = np.unique(results['nns'], return_counts=True)
            for id_,n in zip(ids,counts):
                if id_ in return_ids:
                    return_ids[int(id_)] += n
                else:
                    return_ids[int(id_)] = n

    if save:
        return nn_paths
    else:
        return return_ids




