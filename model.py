import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class rec_model(nn.Module):

    def __init__(self, user_max_dict, movie_max_dict, convParams, embed_dim=32, fc_size=200):
        '''

        Args:
            user_max_dict: the max value of each user attribute. {'uid': xx, 'gender': xx, 'age':xx, 'job':xx}
            user_embeds: size of embedding_layers.
            movie_max_dict: {'mid':xx, 'mtype':18, 'mword':15}
            fc_sizes: fully connect layer sizes. normally 2
        '''

        super(rec_model, self).__init__()

        # --------------------------------- user channel ----------------------------------------------------------------
        # user embeddings
        self.embedding_uid = nn.Embedding(user_max_dict['uid'], embed_dim)
        self.embedding_gender = nn.Embedding(user_max_dict['gender'], embed_dim // 2)
        self.embedding_age = nn.Embedding(user_max_dict['age'], embed_dim // 2)
        self.embedding_job = nn.Embedding(user_max_dict['job'], embed_dim // 2)

        # user embedding to fc: the first dense layer
        self.fc_uid = nn.Linear(embed_dim, embed_dim)
        self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_age = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_job = nn.Linear(embed_dim // 2, embed_dim)

        # concat embeddings to fc: the second dense layer
        self.fc_user_combine = nn.Linear(4 * embed_dim, fc_size)

        # --------------------------------- movie channel -----------------------------------------------------------------
        # movie embeddings
        self.embedding_mid = nn.Embedding(movie_max_dict['mid'], embed_dim)  # normally 32
        self.embedding_mtype_sum = nn.EmbeddingBag(movie_max_dict['mtype'], embed_dim, mode='sum')

        self.fc_mid = nn.Linear(embed_dim, embed_dim)
        self.fc_mtype = nn.Linear(embed_dim, embed_dim)

        # movie embedding to fc
        self.fc_mid_mtype = nn.Linear(embed_dim * 2, fc_size)

        # text convolutional part
        # wordlist to embedding matrix B x L x D  L=15 15 words
        self.embedding_mwords = nn.Embedding(movie_max_dict['mword'], embed_dim)

        # input word vector matrix is B x 15 x 32
        # load text_CNN params
        kernel_sizes = convParams['kernel_sizes']
        # 8 kernel, stride=1,padding=0, kernel_sizes=[2x32, 3x32, 4x32, 5x32]
        self.Convs_text = [nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(k, embed_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(15 - k + 1, 1), stride=(1, 1))
        ).to(device) for k in kernel_sizes]

        # movie channel concat
        self.fc_movie_combine = nn.Linear(embed_dim * 2 + 8 * len(kernel_sizes), fc_size)  # tanh

        # BatchNorm layer
        self.BN = nn.BatchNorm2d(1)

    def forward(self, user_input, movie_input):
        # pack train_data
        uid = user_input['uid']
        gender = user_input['gender']
        age = user_input['age']
        job = user_input['job']

        mid = movie_input['mid']
        mtype = movie_input['mtype']
        mtext = movie_input['mtext']
        if torch.cuda.is_available():
            uid, gender, age, job,mid,mtype,mtext = \
            uid.to(device), gender.to(device), age.to(device), job.to(device), mid.to(device), mtype.to(device), mtext.to(device)
        # user channel
        feature_uid = self.BN(F.relu(self.fc_uid(self.embedding_uid(uid))))
        feature_gender = self.BN(F.relu(self.fc_gender(self.embedding_gender(gender))))
        feature_age =  self.BN(F.relu(self.fc_age(self.embedding_age(age))))
        feature_job = self.BN(F.relu(self.fc_job(self.embedding_job(job))))

        # feature_user B x 1 x 200
        feature_user = F.tanh(self.fc_user_combine(
            torch.cat([feature_uid, feature_gender, feature_age, feature_job], 3)
        )).view(-1,1,200)

        # movie channel
        feature_mid = self.BN(F.relu(self.fc_mid(self.embedding_mid(mid))))
        feature_mtype = self.BN(F.relu(self.fc_mtype(self.embedding_mtype_sum(mtype)).view(-1,1,1,32)))

        # feature_mid_mtype = torch.cat([feature_mid, feature_mtype], 2)

        # text cnn part
        feature_img = self.embedding_mwords(mtext)  # to matrix B x 15 x 32
        flattern_tensors = []
        for conv in self.Convs_text:
            flattern_tensors.append(conv(feature_img.view(-1,1,15,32)).view(-1,1, 8))  # each tensor: B x 8 x1 x 1 to B x 8

        feature_flattern_dropout = F.dropout(torch.cat(flattern_tensors,2), p=0.5)  # to B x 32

        # feature_movie B x 1 x 200
        feature_movie = F.tanh(self.fc_movie_combine(
            torch.cat([feature_mid.view(-1,1,32), feature_mtype.view(-1,1,32), feature_flattern_dropout], 2)
        ))

        output = torch.sum(feature_user * feature_movie, 2)  # B x rank
        return output, feature_user, feature_movie

