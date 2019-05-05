from torch.utils.data import Dataset
import pickle as pkl
import torch
from pandas import DataFrame

class MovieRankDataset(Dataset):

    def __init__(self, pkl_file):

        self.dataFrame = pkl.load(open(pkl_file,'rb'))
    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):

        # user data
        uid = self.dataFrame.ix[idx]['user_id']
        gender = self.dataFrame.ix[idx]['user_gender']
        age = self.dataFrame.ix[idx]['user_age']
        job = self.dataFrame.ix[idx]['user_job']

        # movie data
        mid = self.dataFrame.ix[idx]['movie_id']
        mtype=self.dataFrame.ix[idx]['movie_type']
        mtext=self.dataFrame.ix[idx]['movie_title']

        # target
        rank = torch.FloatTensor([self.dataFrame.ix[idx]['rank']])
        user_inputs = {
            'uid': torch.LongTensor([uid]).view(1,-1),
            'gender': torch.LongTensor([gender]).view(1,-1),
            'age': torch.LongTensor([age]).view(1,-1),
            'job': torch.LongTensor([job]).view(1,-1)
        }

        movie_inputs = {
            'mid': torch.LongTensor([mid]).view(1,-1),
            'mtype': torch.LongTensor(mtype),
            'mtext': torch.LongTensor(mtext)
        }


        sample = {
            'user_inputs': user_inputs,
            'movie_inputs':movie_inputs,
            'target':rank
        }
        return sample
