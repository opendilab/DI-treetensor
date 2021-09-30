# DI-treetensor

[![PyPI](https://img.shields.io/pypi/v/DI-treetensor)](https://pypi.org/project/DI-treetensor/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/DI-treetensor)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/bcda5612b798ebcd354f35447139a4a5/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/bcda5612b798ebcd354f35447139a4a5/raw/comments.json)

[![Docs Deploy](https://github.com/opendilab/DI-treetensor/workflows/Docs%20Deploy/badge.svg)](https://github.com/opendilab/DI-treetensor/actions?query=workflow%3A%22Docs+Deploy%22)
[![Code Test](https://github.com/opendilab/DI-treetensor/workflows/Code%20Test/badge.svg)](https://github.com/opendilab/DI-treetensor/actions?query=workflow%3A%22Code+Test%22)
[![Badge Creation](https://github.com/opendilab/DI-treetensor/workflows/Badge%20Creation/badge.svg)](https://github.com/opendilab/DI-treetensor/actions?query=workflow%3A%22Badge+Creation%22)
[![Package Release](https://github.com/opendilab/DI-treetensor/workflows/Package%20Release/badge.svg)](https://github.com/opendilab/DI-treetensor/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/opendilab/DI-treetensor/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/opendilab/DI-treetensor)

[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-treetensor)](https://github.com/opendilab/DI-treetensor/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-treetensor)](https://github.com/opendilab/DI-treetensor/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/DI-treetensor)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-treetensor)](https://github.com/opendilab/DI-treetensor/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/DI-treetensor)](https://github.com/opendilab/DI-treetensor/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/DI-treetensor)](https://github.com/opendilab/DI-treetensor/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-treetensor)](https://github.com/opendilab/DI-treetensor/blob/master/LICENSE)

`treetensor` is a generalized tree-based tensor structure mainly developed by [OpenDILab Contributors](https://github.com/opendilab).

Almost all the operation can be supported in form of trees in a convenient way to simplify the structure processing when the calculation is tree-based.

## Installation

You can simply install it with `pip` command line from the official PyPI site.

```shell
pip install di-treetensor
```

For more information about installation, you can refer to [Installation](https://opendilab.github.io/DI-treetensor/main/tutorials/installation/index.html#).

## Documentation

The detailed documentation are hosted on [https://opendilab.github.io/DI-treetensor](https://opendilab.github.io/DI-treetensor/).

Only english version is provided now, the chinese documentation is still under development.

## Quick Start

You can easily create a tree value object based on `FastTreeValue`.

```python
import os

import treetensor.torch as torch

if __name__ == '__main__':
    t = torch.randn({'a': (2, 3), 'b': {'x': (3, 4)}})
    print(t)  # tree based tensors

    # some calculations
    print('t ** 2:', t ** 2, sep=os.linesep)
    print('torch.sin(t).cos()', torch.sin(t).cos(), sep=os.linesep)

    t.requires_grad_(True)
    t.std().arctan().backward()
    print('grad of t:', t.grad, sep=os.linesep)  # backward

```

The result should be

```text
<Tensor 0x7fcb33e922b0>
├── a --> tensor([[-0.1105, -1.0873, -1.8016],
│                 [-1.2290,  0.1401, -2.5237]])
└── b --> <Tensor 0x7fcb33e92370>
    └── x --> tensor([[ 0.1579,  0.9740,  0.3076,  0.2183],
                      [ 0.5574,  0.4028, -2.2795,  1.5523],
                      [-0.3870, -1.1649,  0.0314, -0.2728]])

t ** 2:
<Tensor 0x7fcb33e92730>
├── a --> tensor([[0.0122, 1.1822, 3.2456],
│                 [1.5105, 0.0196, 6.3691]])
└── b --> <Tensor 0x7fcb33e92610>
    └── x --> tensor([[2.4920e-02, 9.4863e-01, 9.4633e-02, 4.7653e-02],
                      [3.1067e-01, 1.6226e-01, 5.1961e+00, 2.4097e+00],
                      [1.4976e-01, 1.3571e+00, 9.8400e-04, 7.4414e-02]])

torch.sin(t).cos()
<Tensor 0x7fcb33ead1c0>
├── a --> tensor([[0.9939, 0.6330, 0.5624],
│                 [0.5880, 0.9903, 0.8368]])
└── b --> <Tensor 0x7fcb33ead040>
    └── x --> tensor([[0.9877, 0.6770, 0.9545, 0.9766],
                      [0.8633, 0.9241, 0.7254, 0.5404],
                      [0.9296, 0.6068, 0.9995, 0.9639]])

grad of t:
<Tensor 0x7fcb33f08d60>
├── a --> tensor([[ 0.0060, -0.0174, -0.0345],
│                 [-0.0208,  0.0120, -0.0518]])
└── b --> <Tensor 0x7fcb33f08dc0>
    └── x --> tensor([[ 0.0125,  0.0320,  0.0160,  0.0139],
                      [ 0.0220,  0.0183, -0.0460,  0.0459],
                      [-0.0006, -0.0192,  0.0094,  0.0021]])
```

For more quick start explanation and further usage, take a look at:

* [Quick Start](https://opendilab.github.io/DI-treetensor/main/tutorials/quick_start/index.html)

## Contribution

We appreciate all contributions to improve DI-treetensor, both logic and system designs. Please refer to CONTRIBUTING.md for more guides.

And users can join our [slack communication channel](https://join.slack.com/t/opendilab/shared_invite/zt-v9tmv4fp-nUBAQEH1_Kuyu_q4plBssQ), or contact the core developer [HansBug](https://github.com/HansBug) for more detailed discussion.

## License

`DI-treetensor` released under the Apache 2.0 license.
