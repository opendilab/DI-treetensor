import time
import numpy as np
import torch
import nestedtensor
from ding.torch_utils import Transformer


def generate_variable_sequence(N, M, sample_range):
    return [torch.zeros(1, np.random.randint(*sample_range), M) for _ in range(N)]


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
            max_n = max([d.shape[1] for d in data])
            new_data = torch.zeros(len(data), max_n, data[0].shape[-1]).to(data[0].device)
            mask = torch.zeros(len(data), max_n)
            for i in range(len(data)):
                mask[i, :data[i].shape[1]].add_(1)
            mask = mask.bool().to(data[0].device)

            padding_output = model(new_data, mask=mask)

            output = []
            for i in range(len(data)):
                output.append(padding_output[i, :data[i].shape[1]].unsqueeze(0))
        if cuda:
            torch.cuda.synchronize()
        t_end = time.time()
        result.append(t_end - t_start)
    print('padding_method test time avg: {}, max: {}'.format(np.mean(result), np.max(result)))
    return output, result


def nestedtensor_method(model, data, cuda, test_loop=3):
    raise NotImplementedError("nestedtensor doesn't support chunk op now")
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
    N, M = 64, 128
    sample_range = [32, 64]
    np.random.seed(0)

    data = generate_variable_sequence(N, M, sample_range)
    model = Transformer(input_dim=M)
    print(model)
    if cuda:
        model.cuda()
        data = [d.cuda() for d in data]
    # warm up
    for _ in range(10):
        with torch.no_grad():
            model(data[0])

    naive_output, naive_result = naive_method(model, data, cuda)
    padding_output, padding_result = padding_method(model, data, cuda)
    # nest_output, nest_result = nestedtensor_method(model, data, cuda)


if __name__ == "__main__":
    main(cuda=False)
