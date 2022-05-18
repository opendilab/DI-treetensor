import torch

if __name__ == '__main__':
    # create a native tensor
    t = torch.randn((6, 2, 3))

    # structural operation
    print(torch.stack([t, t, t]))
    print(torch.cat([t, t, t]))
    print(torch.split(t, (1, 2, 3)))

    # math calculations
    print(t ** 2)
    print(torch.sin(t).cos())
