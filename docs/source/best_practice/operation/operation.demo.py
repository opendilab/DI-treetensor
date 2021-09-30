import copy

import torch

import treetensor.torch as ttorch

T, B = 3, 4


def with_nativetensor(batch_):
    mean_b_list = []
    even_index_a_list = []
    for i in range(len(batch_)):
        for k, v in batch_[i].items():
            if k == 'a':
                v = v.float()
                even_index_a_list.append(v[::2])
            elif k == 'b':
                v = v.float()
                transformed_v = torch.pow(v, 2) + 1.0
                mean_b_list.append(transformed_v.mean())
            elif k == 'c':
                for k1, v1 in v.items():
                    if k1 == 'd':
                        v1 = v1.float()
                    else:
                        print('ignore keys: {}'.format(k1))
            else:
                print('ignore keys: {}'.format(k))
    for i in range(len(batch_)):
        for k in batch_[i].keys():
            if k == 'd':
                batch_[i][k]['noise'] = torch.randn(size=(3, 4, 5))

    mean_b = sum(mean_b_list) / len(mean_b_list)
    even_index_a = torch.stack(even_index_a_list, dim=0)
    return batch_, mean_b, even_index_a


def with_treetensor(batch_):
    batch_ = [ttorch.tensor(b) for b in batch_]
    batch_ = ttorch.stack(batch_)
    batch_ = batch_.float()
    batch_.b = ttorch.pow(batch_.b, 2) + 1.0
    batch_.c.noise = ttorch.randn(size=(B, 3, 4, 5))
    mean_b = batch_.b.mean()
    even_index_a = batch_.a[:, ::2]
    batch_ = ttorch.split(batch_, split_size_or_sections=1, dim=0)
    return batch_, mean_b, even_index_a


def get_data():
    return {
        'a': torch.rand(size=(T, 8)),
        'b': torch.rand(size=(6,)),
        'c': {
            'd': torch.randint(0, 10, size=(1,))
        }
    }


if __name__ == "__main__":
    batch = [get_data() for _ in range(B)]
    batch0, mean0, even_index_a0 = with_nativetensor(copy.deepcopy(batch))
    batch1, mean1, even_index_a1 = with_treetensor(copy.deepcopy(batch))
    print(batch0)
    print('\n\n')
    print(batch1)

    assert torch.abs(mean0 - mean1) < 1e-6
    print('mean0 & mean1:', mean0, mean1)
    print('\n')

    assert torch.abs((even_index_a0 - even_index_a1).max()) < 1e-6
    print('even_index_a0:', even_index_a0)
    print('even_index_a1:', even_index_a1)

    assert len(batch0) == B
    assert len(batch1) == B
    assert isinstance(batch1[0], ttorch.Tensor)
    print(batch1[0].shape)
