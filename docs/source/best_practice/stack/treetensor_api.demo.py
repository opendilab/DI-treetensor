import torch

import treetensor.torch as ttorch

B = 4


def get_item():
    return {
        'obs': {
            'scalar': torch.randn(12),
            'image': torch.randn(3, 32, 32),
        },
        'action': torch.randint(0, 10, size=(1,)),
        'reward': torch.rand(1),
        'done': False,
    }


data = [get_item() for _ in range(B)]
# execute `stack` op
data = [ttorch.tensor(d) for d in data]
stacked_data = ttorch.stack(data, dim=0)
# validate
print(stacked_data)
assert stacked_data.obs.image.shape == (B, 3, 32, 32)
assert stacked_data.action.shape == (B, 1)
assert stacked_data.reward.shape == (B, 1)
assert stacked_data.done.shape == (B,)
assert stacked_data.done.dtype == torch.bool
