from typing import Sequence, Mapping, Union
import re
import torch
from torch._six import string_classes
import collections.abc as container_abcs
from ctools.pysc2.lib.static_data import ACTIONS_REORDER_INV
from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK

int_classes = int
np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def as_learner_collate_fn(batch):
    def default_collate(batch: Sequence,
                        cat_1dim: bool = True,
                        ignore_prefix: list = ['collate_ignore']) -> Union[torch.Tensor, Mapping, Sequence]:
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, directly concatenate into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            if elem.shape == (1, ) and cat_1dim:
                # reshape (B, 1) -> (B)
                return torch.cat(batch, 0, out=out)
                # return torch.stack(batch, 0, out=out)
            else:
                return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                return default_collate([torch.as_tensor(b) for b in batch], cat_1dim=cat_1dim)
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, int_classes):
            dtype = torch.bool if isinstance(elem, bool) else torch.int64
            return torch.tensor(batch, dtype=dtype)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            ret = {}
            for key in elem:
                if any([key.startswith(t) for t in ignore_prefix]):
                    ret[key] = [d[key] for d in batch]
                else:
                    ret[key] = default_collate([d[key] for d in batch], cat_1dim=cat_1dim)
            return ret
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate(samples, cat_1dim=cat_1dim) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [default_collate(samples, cat_1dim=cat_1dim) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

    def lists_to_dicts(data, recursive=False):
        if len(data) == 0:
            raise ValueError("empty data")
        if isinstance(data[0], dict):
            if recursive:
                new_data = {}
                for k in data[0].keys():
                    if isinstance(data[0][k], dict):
                        tmp = [data[b][k] for b in range(len(data))]
                        new_data[k] = lists_to_dicts(tmp)
                    else:
                        new_data[k] = [data[b][k] for b in range(len(data))]
            else:
                new_data = {k: [data[b][k] for b in range(len(data))] for k in data[0].keys()}
        elif isinstance(data[0], tuple) and hasattr(data[0], '_fields'):  # namedtuple
            new_data = type(data[0])(*list(zip(*data)))
        else:
            raise TypeError("not support element type: {}".format(type(data[0])))
        return new_data

    def sequence_mask(lengths, max_len=None):
        if len(lengths.shape) == 1:
            lengths = lengths.unsqueeze(dim=1)
        bz = lengths.numel()
        if max_len is None:
            max_len = lengths.max()
        else:
            max_len = min(max_len, lengths.max())
        return torch.arange(0, max_len).type_as(lengths).repeat(bz, 1).lt(lengths).to(lengths.device)

    ret = {}
    ret['batch_size'] = batch_size = len(batch)
    batch = list(zip(*batch))
    ret['traj_len'] = traj_len = len(batch)

    ret['prev_state'] = [d.pop('prev_state') for d in batch[0]]

    obs_home_next = [d.pop('obs_home_next') for d in batch[-1]]
    obs_away_next = [d.pop('obs_away_next') for d in batch[-1]]
    new_batch = []
    for s in range(len(batch)):
        new_batch += batch[s]
    new_batch = lists_to_dicts(new_batch)

    new_batch['obs_home'] += obs_home_next
    new_batch['obs_away'] += obs_away_next
    obs = new_batch['obs_home'] + new_batch['obs_away']
    obs = lists_to_dicts(obs)
    if 'actions' in obs.keys():
        obs.pop('actions')

    for k in ['spatial_info', 'scalar_info']:
        ret[k] = default_collate(obs[k])

    entity_raw = lists_to_dicts(obs['entity_raw'])
    entity_raw['location'] = torch.nn.utils.rnn.pad_sequence(entity_raw['location'], batch_first=True)
    ret['entity_raw'] = entity_raw
    ret['entity_num'] = torch.LongTensor([[i.shape[0]] for i in obs['entity_info']])
    ret['entity_info'] = torch.nn.utils.rnn.pad_sequence(obs['entity_info'], batch_first=True)
    ret['map_size'] = obs['map_size']

    ret['selected_units_num'] = torch.stack(new_batch['selected_units_num'], dim=0)
    max_selected_units_num = ret['selected_units_num'].max()

    actions = lists_to_dicts(new_batch['actions'])
    actions_mask = {k: [] for k in actions.keys()}
    for i in range(len(actions['action_type'])):
        action_type = actions['action_type'][i].item()
        flag = action_type == 0
        inv_action_type = ACTIONS_REORDER_INV[action_type]
        actions_mask['action_type'].append(False) if flag else actions_mask['action_type'].append(True)
        actions_mask['delay'].append(False) if flag else actions_mask['delay'].append(True)
        for k in ['queued', 'target_units', 'selected_units', 'target_location']:
            if flag or not GENERAL_ACTION_INFO_MASK[inv_action_type][k]:
                actions_mask[k].append(False)
            else:
                actions_mask[k].append(True)

    for k in actions_mask.keys():
        actions_mask[k] = torch.BoolTensor(actions_mask[k])

    map_size = list(zip(*obs['map_size']))
    assert len(set(map_size[0])) == 1 and len(set(map_size[1])) == 1, 'only support same size map'
    map_size = obs['map_size'][0]
    for k, v in actions.items():
        if k in ['action_type', 'delay', 'repeat', 'queued', 'target_units']:
            actions[k] = torch.cat(v, dim=0)
        elif k == 'target_location':
            actions[k] = torch.stack(v)
            actions[k] = actions[k][:, 1] * map_size[1] + actions[k][:, 0]
            actions[k] = actions[k].long()
        else:
            actions[k] = torch.nn.utils.rnn.pad_sequence(actions[k], batch_first=True)
            actions[k] = actions[k][:, :max_selected_units_num]

    ret['actions'] = actions

    ret['reward'] = {}
    reward = default_collate(new_batch['reward'])
    for k, v in reward.items():
        ret['reward'][k] = v.view(traj_len, batch_size)

    ret['game_second'] = torch.LongTensor(new_batch['game_second'])

    home_size = len(ret['game_second'])
    max_entity_num = ret['entity_num'][:home_size].max()
    for k in ['behaviour_output', 'teacher_output']:
        data = lists_to_dicts(new_batch[k])
        for _k in data.keys():
            if _k in ['action_type', 'delay', 'repeat', 'queued', 'target_location']:
                data[_k] = default_collate(data[_k])
            elif _k == 'selected_units':
                for i in range(len(data[_k])):
                    if len(data[_k][i].shape) == 1:
                        data[_k][i] = data[_k][i].unsqueeze(0)
                    data[_k][i] = data[_k][i][:max_entity_num + 1]
                    data[_k][i] = torch.nn.functional.pad(
                        data[_k][i], (0, max_entity_num + 1 - data[_k][i].shape[1]), 'constant', 0
                    )

                data[_k] = torch.nn.utils.rnn.pad_sequence(data[_k], batch_first=True)
                data[_k] = data[_k][:, :max_selected_units_num]
            elif _k == 'target_units':
                data[_k] = torch.nn.utils.rnn.pad_sequence(data[_k], batch_first=True)
                data[_k] = data[_k][:, :max_entity_num]
        ret[k] = data

    mask = {}
    mask['actions_mask'] = actions_mask
    mask['selected_units_mask'] = sequence_mask(ret['selected_units_num'][:home_size])
    entity_num = ret['entity_num']
    mask['target_units_logits_mask'] = sequence_mask(entity_num[:home_size])
    plus_entity_num = entity_num + 1  # selected units head have one more end embedding
    mask['selected_units_logits_mask'] = sequence_mask(plus_entity_num[:home_size])
    ret['mask'] = mask
    return ret
