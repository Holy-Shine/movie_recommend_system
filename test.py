
from model import rec_model
import torch

uid=torch.tensor(
    [1,2,3,4]
).long


user_max_dict={
    'uid':10,
    'gender':2,
    'age':10,
    'job':10
}

movie_max_dict={
    'mid':10,
    'mtype':20,
    'mword':20
}

convParams={
    'kernel_sizes':[2,3,4,5]
}
def simulatorData():
    uid = mid = age = job = torch.tensor(
        [1, 2, 3, 4]
    ).view(4, -1).long()

    gender = torch.tensor(
        [1, 0, 1, 0]
    ).view(4, -1).long()

    user_inputs = {
        'uid': uid,
        'gender': gender,
        'age': age,
        'job': job
    }

    mtype = torch.tensor(
        [[1] * 18, [2] * 18, [3] * 18, [4] * 18]
    ).long()

    mword = torch.tensor(
        [[1] * 15, [2] * 15, [3] * 15, [4] * 15]
    )

    movie_inputs = {
        'mid': mid,
        'mtype': mtype,
        'mtext': mword
    }
    return (user_inputs, movie_inputs)

if __name__=='__main__':
    model = rec_model(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, convParams=convParams)
    user_inputs, movie_inputs = simulatorData()
    model(user_inputs, movie_inputs)