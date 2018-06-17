# Recommendation Interface

import torch
from torch.utils.data import DataLoader
from dataset import MovieRankDataset

import numpy as np
import pickle as pkl

def saveMovieAndUserFeature(model):
    '''
    Save Movie and User feature into HD

    '''

    datasets = MovieRankDataset(pkl_file='data.p')
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False)

    # format: {id(int) : feature(numpy array)}
    user_feature_dict = {}
    movie_feature_dict = {}
    movies=set()
    users = set()
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(dataloader):
            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']

            # B x 1 x 1 = 1 x 1 x 1
            uid = user_inputs['uid'].item()   # uid
            mid = movie_inputs['mid'].item()  # mid

            movies.add(mid)
            users.add(uid)

            # B x 1 x 200 = 1 x 1 x 200
            _, feature_user, feature_movie = model(user_inputs, movie_inputs)

            # 1 x 200
            feature_user = feature_user.view(-1,200).numpy()
            feature_movie = feature_movie.view(-1,200).numpy()

            if uid not in user_feature_dict.keys():
                user_feature_dict[uid]=feature_user
            if mid not in movie_feature_dict.keys():
                movie_feature_dict[mid]=feature_movie

    feature_data = {'feature_user': feature_user, 'feature_movie':feature_movie}
    ids_user_movie={'user': users, 'movie':movies}
    pkl.dump(feature_data,open('Params/feature_data.pkl','wb'))
    pkl.dump(ids_user_movie, open('Params/user_movie_ids.pkl','wb'))



def getKNNitem(itemID,itemName='movie',K=1):
    '''
    Use KNN at feature data to get K neighbors

    Args:
        itemID: target item's id
        itemName: 'movie' or 'user'
        K: K-neighbors

    return:
        a list of item ids of which close to itemID
    '''
    assert K>=1, 'Expect K bigger than 0 but get K<1'

    # get cosine similarity between vec1 and vec2
    def getCosineSimilarity(vec1, vec2):
        cosine_sim = float(vec1.dot(vec2.T).item()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_sim

    feature_data = pkl.load(open('Params/feature_data.pkl','rb'))

    feature_items = feature_data['itemName']
    feature_current = feature_items[itemID]

    id_sim = [(item_id,getCosineSimilarity(feature_current,vec2)) for item_id,vec2 in feature_items.items()]
    id_sim = sorted(id_sim,key=lambda x:x[1],reverse=True)

    return [id_sim[i][0] for i in range(K+1)][1:]


def getUserMostLike(model,uid):
    '''
    Get user(uid) mostly like movie

    Args:
        model: net model
        uid: target user's id

    return:
        the biggest rank movie id
    '''

    user_movie_ids = pkl.load(open('Params/user_movie_ids.pkl','rb'))
    movie_ids = user_movie_ids['movie']

    mid_rank={}

    # Step 1. Go through net to get user_movie score
    user_inputs = torch.LongTensor([uid]).view(-1,1,1)
    with torch.no_grad():
        for mid in movie_ids:
            movie_inputs = torch.LongTensor([mid]).view(-1,1,1)

            rank, _, _ = model(user_inputs,movie_inputs)

            if mid not in mid_rank.keys():
                mid_rank[mid]=rank.item()

    mid_rank = [(mid, rank) for mid, rank in mid_rank.items()]
    mids = [mid[0] for mid in sorted(mid_rank, key=lambda x: x[1], reverse=True)]

    return mids[0]




