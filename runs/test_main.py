import treetensor.torch as torch

if __name__ == '__main__':
    # create a tree tensor
    t = torch.randn({
        'a': (6, 2, 3),
        'b': {'x': (6, 3), 'y': (6, 1, 4)},
    })

    # structural operation
    print(torch.stack([t, t, t]))
    print(torch.cat([t, t, t]))
    print(torch.split(t, (1, 2, 3)))

    # math calculations
    print(t ** 2)
    print(torch.sin(t).cos())
