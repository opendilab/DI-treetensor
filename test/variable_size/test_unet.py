import time
import numpy as np
import torch
import nestedtensor
from unet import UNet


def same_test(model, data, cuda, test_loop=3):
    result = []
    new_data = torch.cat([data[0].clone() for _ in range(len(data))], dim=0)
    for _ in range(test_loop):
        t_start = time.time()
        with torch.no_grad():
            output = model(new_data)
        if cuda:
            torch.cuda.synchronize()
        t_end = time.time()
        result.append(t_end - t_start)
    print('same_shape test time avg: {}, max: {}'.format(np.mean(result), np.max(result)))
    return output, result


def naive_method(model, data, cuda, test_loop=3):
    result = []
    for _ in range(test_loop):
        t_start = time.time()
        with torch.no_grad():
            output = [model(d) for d in data]
        if cuda:
            torch.cuda.synchronize()
        t_end = time.time()
        result.append(t_end - t_start)
    print('naive_method test time avg: {}, max: {}'.format(np.mean(result), np.max(result)))
    return output, result


def padding_method(model, data, cuda, test_loop=3):
    result = []
    for _ in range(test_loop):
        t_start = time.time()
        with torch.no_grad():
            # padding
            max_h = max([d.shape[-2] for d in data])
            max_w = max([d.shape[-1] for d in data])
            new_data = torch.zeros(len(data), 3, max_h, max_w).to(data[0].device)
            start_h = [max_h - d.shape[-2] for d in data]
            start_w = [max_w - d.shape[-1] for d in data]
            for i in range(len(data)):
                new_data[i, :, start_h[i]:, start_w[i]:] = data[i]

            padding_output = model(new_data)

            output = []
            for i in range(len(data)):
                output.append(
                    padding_output[i, :, start_h[i]:start_h[i] + data[i].shape[-2],
                                   start_w[i]:start_w[i] + data[i].shape[-1]]
                )
        if cuda:
            torch.cuda.synchronize()
        t_end = time.time()
        result.append(t_end - t_start)
    print('padding_method test time avg: {}, max: {}'.format(np.mean(result), np.max(result)))
    return output, result


def nestedtensor_method(model, data, cuda, test_loop=3):
    result = []
    data = nestedtensor.nested_tensor([d.squeeze(0) for d in data])
    for _ in range(test_loop):
        t_start = time.time()
        with torch.no_grad():
            output = model(data)
        if cuda:
            torch.cuda.synchronize()
        t_end = time.time()
        result.append(t_end - t_start)
    print('nestedtensor_method test time avg: {}, max: {}'.format(np.mean(result), np.max(result)))
    output = [o.unsqueeze(0) for o in output]
    return output, result


def main(cuda):
    B, H, W = 8, 144, 168
    S = 8
    model = UNet()
    print(model)
    data = [
        torch.randn(1, 3, H, W),
        torch.randn(1, 3, H + S, W + S),
        torch.randn(1, 3, H - S, W),
        torch.randn(1, 3, H, W - S),
        torch.randn(1, 3, H, W),
        torch.randn(1, 3, H + S, W + S),
        torch.randn(1, 3, H - S, W),
        torch.randn(1, 3, H, W - S),
    ]
    if cuda:
        model.cuda()
        data = [d.cuda() for d in data]
    # warm up
    for _ in range(10):
        model(data[0])

    same_output, same_result = same_test(model, data, cuda)
    naive_output, naive_result = naive_method(model, data, cuda)
    assert len(naive_output) == B, len(naive_output)
    padding_output, padding_result = padding_method(model, data, cuda)
    nest_output, nest_result = nestedtensor_method(model, data, cuda)
    print(naive_output[0][0, 0, 0, :10])
    print(nest_output[0][0, 0, 0, :10])


if __name__ == "__main__":
    main(cuda=False)
