import torch
from ctools.pysc2.lib.static_data import ACTIONS_REORDER_INV
from ctools.pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK
from treevalue import mapping, union, TreeValue
import treetensor.torch as ttorch
from treetensor.torch import Padding1D, Padding2D


def as_learner_collate_fn(batch):
    ret = ttorch.Tensor({})
    ret.batch_size = batch_size = len(batch)
    batch = list(zip(*batch))  # TODO
    ret.traj_len = traj_len = len(batch)

    batch = [[ttorch.Tensor(TreeValue(j)) for j in i] for i in batch]
    batch = union(*batch, mode='outer', missing=None)
    batch = mapping(batch, lambda x: sum(x, []))
    ret.prev_state = batch.prev_state[:batch_size]

    obs_home_next = batch.obs_home_next[-batch_size:]
    obs_away_next = batch.obs_away_next[-batch_size:]

    batch.obs_home += obs_home_next
    batch.obs_away += obs_away_next
    obs = batch.obs_home + batch.obs_away
    if 'actions' in obs:
        print('pop actions')
        obs.pop('actions')  # TODO

    ret.spatial_info = ttorch.stack(obs.spatial_info)
    ret.scalar_info = ttorch.stack(obs.scalar_info)

    ret.entity_info, _, ret.entity_ori_shape = Padding1D(obs.entity_info)
    ret.entity_num = torch.as_tensor([t[0] for t in ret.entity_ori_shape])  # entity_num shape less 1 dim
    obs.entity_raw.location = Padding1D(obs.entity_raw.location)[0]
    ret.entity_raw = obs.entity_raw
    ret.map_size = obs.map_size

    ret.selected_units_num = ttorch.stack(batch.selected_units_num)
    max_selected_units_num = ret.selected_units_num.max()
    # map_size = list(zip(*obs['map_size']))  # TODO
    # assert len(set(map_size[0])) == 1 and len(set(map_size[1])) == 1, 'only support same size map'
    map_size = ret.map_size[0]
    ret.reward = ttorch.stack(batch.reward).view(traj_len, batch_size)
    ret.game_second = torch.as_tensor(batch.game_second).long()  # tricky
    home_size = len(ret.game_second)

    actions = batch.actions
    actions.selected_units, selected_units_mask, _ = Padding1D(actions.selected_units)
    actions.selected_units = actions.selected_units[:, :max_selected_units_num]
    actions = ttorch.stack(actions)  # target_units shape more 1 dim
    actions.target_location = (actions.target_location[..., 1] * map_size[1] + actions.target_location[..., 0]).long()
    ret.actions = actions

    action_type = actions.action_type.squeeze(-1)
    flag = action_type == 0
    action_type_list = action_type.tolist()
    inv_action_type = [ACTIONS_REORDER_INV[t] for t in action_type_list]
    actions_mask = [GENERAL_ACTION_INFO_MASK[t] for t in inv_action_type]
    actions_mask = [{k: [v] for k, v in t.items() if k in ['queued', 'target_location', 'target_units', 'selected_units']} for t in actions_mask]
    actions_mask = ttorch.cat([ttorch.Tensor(t) for t in actions_mask])
    actions_mask.action_type = flag  # whether deepcopy
    actions_mask.delay = flag
    actions_mask = actions_mask.bool()
    mask = ttorch.Tensor({'actions_mask': actions_mask, 'selected_units_mask': selected_units_mask[:, :max_selected_units_num]})

    max_entity_num = ret.entity_num[:home_size].max()
    behaviour_output_sel, selected_units_logits_mask, _ = Padding2D(batch.behaviour_output.selected_units)
    batch.behaviour_output.selected_units = behaviour_output_sel[:, :max_selected_units_num, :max_entity_num + 1]
    mask.selected_units_logits_mask = selected_units_logits_mask[:, 0, :max_entity_num + 1]
    batch.teacher_output.selected_units = Padding2D(batch.teacher_output.selected_units)[0][:, :max_selected_units_num, :max_entity_num + 1]

    behaviour_output_tar, target_units_logits_mask, _ = Padding1D(batch.behaviour_output.target_units)
    batch.behaviour_output.target_units = behaviour_output_tar[:, :max_entity_num]
    mask.target_units_logits_mask = target_units_logits_mask[:, :max_entity_num]
    batch.teacher_output.target_units = Padding1D(batch.teacher_output.target_units)[0][:, :max_entity_num]
    ret.behaviour_output = ttorch.stack(batch.behaviour_output)
    ret.teacher_output = ttorch.stack(batch.teacher_output)

    ret.mask = mask
    return ret
