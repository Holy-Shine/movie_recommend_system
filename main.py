from model import rec_model
from dataset import MovieRankDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --------------- hyper-parameters------------------
user_max_dict={
    'uid':6041,  # 6040 users
    'gender':2,
    'age':7,
    'job':21
}

movie_max_dict={
    'mid':3953,  # 3952 movies
    'mtype':18,
    'mword':5215   # 5215 words
}

convParams={
    'kernel_sizes':[2,3,4,5]
}


def train(model,num_epochs=5, lr=0.1):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    datasets = MovieRankDataset(pkl_file='data.p')
    dataloader = DataLoader(datasets,batch_size=256,shuffle=True)

    losses=[]
    for epoch in range(num_epochs):
        loss_all = 0
        for i_batch,sample_batch in enumerate(dataloader):

            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            target = sample_batch['target'].to(device)

            model.zero_grad()

            tag_rank = model(user_inputs, movie_inputs)

            loss = loss_function(tag_rank, target)
            if i_batch%100 ==0:
                print('loss after 100 batches{}'.format(loss))
            loss_all += loss
            loss.backward()
            optimizer.step()
        print('Epoch {}:\t loss:{}'.format(epoch,loss_all))
if __name__=='__main__':
    model = rec_model(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, convParams=convParams)
    model=model.to(device)
    train(model=model)