'''
ivan
code is from  rdm
根据给定的text, 找到与之相关的  k个img_embeddings
config : 有 image_embeddings  的路径  ,有text_embeddings 路径
save_dir :保存的路径
(查看)find_nns .yaml文件
'''
import os
import torch
import pickle
import sys
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm
from Utils.ldm_utils import  instantiate_from_config
from argparse import ArgumentParser
from omegaconf import OmegaConf
import numpy as np

def save_pkl(filepath, save_it, results, corrupts, i, j, start_id, dset_batch_size):
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                old_one = pickle.load(f)

            old_one.update({results: save_it[results]})
            with open(filepath, 'wb') as f:
                pickle.dump(old_one, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'ERROR: {e.__class__.__name__} : ', e)
            if npatches_perside == 1:
                print(f'Overwriting id {start_id + i * dset_batch_size + j} as it is corrupt.')
                with open(filepath, 'wb') as f:
                    pickle.dump(save_it, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                corrupts.add(start_id + i * dset_batch_size + j)
                print(f'Adding id {start_id + i * dset_batch_size + j} to corrupts.')
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(save_it, f, protocol=pickle.HIGHEST_PROTOCOL)

    return corrupts


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
        query=batch['embeddings'].to(device)
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
            # 就是results应该是 [batch,embeddings, 然后是k 个 embeddings 以及对应的ids
            #
            for j in range(len(results['embeddings'])):
                # 保存 同一个 batch 中的每一个， 这里不要动 ！！
                filename = f'embeddings/{dataset_builder.k}_nns-img{start_id + i * dset_batch_size + j:09d}.p'
                filepath = os.path.join(base_savedir, filename)
                save_it = {results: {
                    'query':query,
                    'embeddings': results['embeddings'][j],
                    'img_ids': results['img_ids'][j],
                    'nn_ids': results['nns'][j]
                }}
                corrupts = save_pkl(filepath=filepath,
                                    save_it=save_it,
                                    results=results,
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

def get_parser():
    parser = ArgumentParser()

    parser.add_argument('-c', '--config',
                        required=True,
                        type=str,
                        help='config containing dataset to load, retrieval feature extractor and metric to compute similarities')

    parser.add_argument('-v', '--visualize',
                        default=False,
                        action='store_true',
                        help='Start builder in visualization mode?')

    parser.add_argument('-r', '--random_seed',
                        default=None,
                        type=int,
                        help='Random seed')


    return parser

if __name__ == '__main__':
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt = parser.parse_args()

    if opt.only_patches and opt.only_neighbors:
        raise ValueError('both --only_patches and --only_neighbors options are selected')


    config = OmegaConf.load(opt.rconfig)
    query_config = OmegaConf.load(opt.qconfig).data
    query_config = query_config.params[opt.split]

    # build retrieval dataset and train searcher
    dataset_builder: DatasetBuilder = instantiate_from_config(config.builder)
    #  class : dsetbuilder.DatasetBuilder
    # setting some paths
    base_savedir = os.path.join(
        opt.nns_savedir,
        f'{dataset_builder.data_pool["embedding"].shape[0]}p-{dataset_builder.retriever_name}_{dataset_builder.patch_size}@{dataset_builder.dset_name}',
        f'{query_config.params.dset_config.target.split(".")[-1]}',
    )

    print(f'Base savedir is {base_savedir}')
    #  Base savedir is neighbors/9999990p-ClipImageRetriever_256@FullOpenImagesTrain/ImageNetValidation
    assert opt.parts >= 1, 'Specified number of parts for the subset must be greater or equal 1'
    cfile_name = 'corrupts'
    mfile_name =  'nn_paths'


    mfile_name += '.p'
    cfile_name += '.txt'
    meta_filepath = os.path.join(base_savedir, mfile_name)
    c_filepath = os.path.join(base_savedir, cfile_name)

    if opt.mode == 'text':
        print('*'*100)
        print('Setting n_patches per side to 1 as other options not supported until now')
        print('*' * 100)
        opt.log_max_np = 0

    if not opt.only_patches:
        dataset_builder.build_data_pool()
        dataset_builder.load_embeddings()
        dataset_builder.train_searcher()
        nns = {}
        id_key = 0
        corrupts = set()

        device = next(dataset_builder.retriever.parameters()).device
        nn_paths = {}
        for n_p in range(opt.log_max_np+1):
            npatches_perside = 2**n_p
            n_patches = npatches_perside ** 2
            print(f'computing {dataset_builder.k} nns for {n_patches} patches per Image.')
            query_config.params['n_patches_per_side'] = npatches_perside
            query_dataset = instantiate_from_config(query_config)

            embeddings_savedir = os.path.join(base_savedir, 'embeddings')
            os.makedirs(base_savedir, exist_ok=True)
            os.makedirs(embeddings_savedir,exist_ok=True)
            dset_batch_size = opt.batch_size // n_patches
            qloader = DataLoader(
                query_dataset,
                batch_size=dset_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=opt.n_workers
            )
            nn_paths= search_nns(
                dataset_builder=dataset_builder,
                qloader=qloader,
                mode=opt.mode,
                save=True,
                npatches_perside=npatches_perside,
                base_savedir=base_savedir,
                nn_paths=nn_paths,
                corrupts=corrupts,
                start_id=start_id,
                device=device,
            )
            with open(meta_filepath, 'wb') as f:
                # 总之相当于将 之前的信息进行保存
                pickle.dump(nn_paths, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(c_filepath,'w') as fc:
            for item in corrupts:
                fc.write(str(item)+'\n')
    if (opt.parts == 1 or opt.only_patches) and not opt.only_neighbors:
        print(f'loading precomputed nns from "{meta_filepath}"')
        assert os.path.isfile(meta_filepath)
        with open(meta_filepath,'rb') as f:
            nn_paths = pickle.load(f)
        print('done')

