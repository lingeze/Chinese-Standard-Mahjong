"""
battle.py — SL 模型对战评估

使用 RL/env.py 的游戏模拟器，加载 SL 训练的模型进行多局对战，
统计各项指标并输出对战报告。

支持不同网络结构的模型混战，通过 --model-classes 按模型指定类。

用法示例:
    # 2模型对战（默认都用 model.CNNModel），500局
    python battle.py --models a.pkl b.pkl --games 500

    # 4个不同模型各坐1个位置
    python battle.py --models m0.pkl m1.pkl m2.pkl m3.pkl --games 200

    # 指定座位
    python battle.py --models a.pkl b.pkl --seats 0 1 0 1 --games 100

    # ★ 不同网络结构的模型对战（核心功能）
    python battle.py \
        --models log/baseline/19.pkl log/resnet/19.pkl \
        --model-classes model.CNNModel model.ResNetModel \
        --games 500

    # 自对弈
    python battle.py --models baseline.pkl --selfplay --games 500

    # 指定设备
    python battle.py --models a.pkl b.pkl --games 500 --device cuda
"""

import sys
import os
import time
import csv
import json
import argparse
import importlib
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

# ── 将 RL 目录加入 path，以便导入其 env / feature / agent ──
_RL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RL')
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)

from env import MahjongGBEnv
from feature import FeatureAgent as RLFeatureAgent

# ── 将 SL 目录加入 path，以便默认导入 model ──
_SL_DIR = os.path.dirname(os.path.abspath(__file__))
if _SL_DIR not in sys.path:
    sys.path.insert(0, _SL_DIR)


# ═══════════════════════════════════════════════════════════════════════
# 动态模型类导入
# ═══════════════════════════════════════════════════════════════════════

def import_model_class(class_spec: str):
    """
    根据 "module.ClassName" 字符串动态导入模型类。

    对于不带包前缀的简单模块名（如 "model"），显式从 SL 目录加载，
    避免 import 时误加载 RL/model.py。
    """
    importlib.invalidate_caches()

    parts = class_spec.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(
            f"无法解析模型类 '{class_spec}'。"
            f"格式应为 'module.ClassName'，如 'model.CNNModel'。"
        )
    module_name, class_name = parts

    # ── 简单模块名（无包前缀）：显式从 SL 目录加载 ──
    if '.' not in module_name:
        sl_path = os.path.join(_SL_DIR, module_name + '.py')
        if os.path.exists(sl_path):
            spec = importlib.util.spec_from_file_location(module_name, sl_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod   # 注册到全局缓存，防止后续再次导入冲突
            spec.loader.exec_module(mod)
            print(f'    → module "{module_name}" loaded from: {sl_path}')
        else:
            mod = importlib.import_module(module_name)
            print(f'    → module "{module_name}" loaded from: {getattr(mod, "__file__", "unknown")}')
    else:
        mod = importlib.import_module(module_name)
        print(f'    → module "{module_name}" loaded from: {getattr(mod, "__file__", "unknown")}')

    cls = getattr(mod, class_name, None)
    if cls is None:
        available = sorted(
            name for name in dir(mod)
            if not name.startswith('_')
            and isinstance(getattr(mod, name), type)
            and issubclass(getattr(mod, name), torch.nn.Module)
        )
        raise ImportError(
            f"模块 '{module_name}' 中没有找到类 '{class_name}'。\n"
            f"  可用的 nn.Module 子类: {available if available else '(无)'}"
        )
    return cls


# ═══════════════════════════════════════════════════════════════════════
# ModelWrapper — 将 SL 模型适配到 RL env 的接口
# ═══════════════════════════════════════════════════════════════════════

class ModelWrapper:
    """
    封装任意 SL 训练出的模型（不限于 CNNModel），使其可直接用于 RL 的 MahjongGBEnv。

    模型接口约定（forward）:
        输入  -> {'is_training': bool, 'obs': {'observation': (B,C,4,9), 'action_mask': (B,235)}}
        输出  -> (B, 235)  logits（已 masked）

    RL env:
        产出  -> {'observation': ndarray(C,4,9), 'action_mask': ndarray(235,)}
        期望  -> action_index (int)
    """

    def __init__(self, model_class, checkpoint_path, device='cpu'):
        self.device = device
        self.model = model_class()   # 调用无参构造函数
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.train(False)
        self.model.to(device)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())

    @property
    def model_class_name(self):
        return type(self.model).__name__

    def __call__(self, obs):
        """
        obs: {'observation': ndarray (C,4,9), 'action_mask': ndarray (235,)}
        返回: action_index (int)
        """
        obs_t = torch.from_numpy(
            np.asarray(obs['observation'], dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(
            np.asarray(obs['action_mask'], dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        input_dict = {
            'is_training': False,
            'obs': {'observation': obs_t, 'action_mask': mask_t},
        }

        with torch.no_grad():
            logits = self.model(input_dict)

        return logits.flatten().argmax().item()


# ═══════════════════════════════════════════════════════════════════════
# 单局对战
# ═══════════════════════════════════════════════════════════════════════

def run_one_game(env, models, seat_to_model_idx, verbose=False, capture_invalid=False):
    """
    运行一局完整的 4 人麻将。

    参数:
        env: MahjongGBEnv 实例
        models: list[ModelWrapper]，按模型索引访问
        seat_to_model_idx: list[int]，长度 4，seat_to_model_idx[seat] = 该座位使用的模型索引
        verbose: 打印每步动作
        capture_invalid: 当发生无效动作时，记录详细的局面信息

    返回:
        dict: {
            'winner': 赢家座位 (0-3)，流局/无效则为 -1,
            'is_self_drawn': bool,
            'deal_in_seat': 点炮者座位 (-1 表示无),
            'fan_count': 番数 (0 表示流局/无效),
            'scores': [4],      # 按座位
            'seat_models': [4], # 每座位的模型索引
            'total_steps': int,
            'invalid_detail': dict or None,  # 仅当 capture_invalid=True 且本局无效时
        }
    """
    obs = env.reset()
    done = False
    steps = 0
    last_action = {}  # agent_name -> (action_index, response_string)
    last_action_mask = {}  # agent_name -> action_mask_array (only when capture_invalid)

    while not done:
        actions = {}
        for agent_name, agent_obs in obs.items():
            seat = int(agent_name.split('_')[1]) - 1       # 0-indexed
            mid = seat_to_model_idx[seat]
            action = models[mid](agent_obs)
            actions[agent_name] = action

            agent = env.agents[seat]
            resp = agent.action2response(action)
            last_action[agent_name] = (action, resp, seat, mid)

            # 诊断：记录动作选取时的 mask 状态
            if capture_invalid:
                mask = agent_obs.get('action_mask')
                if mask is not None:
                    last_action_mask[agent_name] = np.asarray(mask).copy()

            if verbose:
                print(f'  [Step {steps}] Seat {seat} (model {mid}): {resp}')

        next_obs, rewards, done = env.step(actions)
        obs = next_obs
        steps += 1

    # ── 解析结果 ──
    result = {
        'winner': -1,
        'is_self_drawn': False,
        'deal_in_seat': -1,
        'fan_count': 0,
        'scores': [0, 0, 0, 0],
        'seat_models': list(seat_to_model_idx),
        'total_steps': steps,
    }

    if rewards is None:
        return result

    score_list = [rewards.get(f'player_{s+1}', 0) for s in range(4)]
    result['scores'] = score_list

    max_score = max(score_list)
    min_score = min(score_list)

    # 无效动作检测必须放在最前面：offender=-30, 其他3家=+10
    # max_score=10 唯一标识无效局（正常和牌 max_score >= 24+1=25）
    if min_score == -30 and max_score == 10:
        result['winner'] = score_list.index(min_score)
        result['fan_count'] = -1
    elif max_score > 0:
        winner = score_list.index(max_score)
        result['winner'] = winner

        # 自摸: 3家等负 → reward[winner] = (8 + fan) * 3
        other_scores = [s for i, s in enumerate(score_list) if i != winner]
        if min(other_scores) == max(other_scores) and max(other_scores) < 0:
            result['is_self_drawn'] = True
            result['fan_count'] = max_score // 3 - 8
        else:
            # 点炮: 负分最大的为点炮者 → reward[winner] = 24 + fan
            result['is_self_drawn'] = False
            for s in range(4):
                if s != winner and score_list[s] == min_score:
                    result['deal_in_seat'] = s
                    break
            result['fan_count'] = max_score - 24
    elif max_score == 0 and min_score == 0:
        # 流局
        pass

    # ── 捕获无效动作的详细局面 ──
    if capture_invalid and result['fan_count'] == -1:
        offender_seat = result['winner']  # 在无效局中 winner 字段存的是犯规者座位
        detail = {
            'game_step': steps,
            'offender_seat': offender_seat,
            'offender_model_idx': seat_to_model_idx[offender_seat],
            'env_state': getattr(env, 'state', -1),
            'cur_player': getattr(env, 'curPlayer', -1),
            'cur_tile': getattr(env, 'curTile', ''),
            'wall_last': getattr(env, 'wallLast', False),
            'prevalent_wind': getattr(env, 'prevalentWind', -1),
        }

        # 犯规者最后尝试的动作
        agent_name = f'player_{offender_seat + 1}'
        if agent_name in last_action:
            act_idx, resp_str, _seat, _mid = last_action[agent_name]
            detail['attempted_action_idx'] = act_idx
            detail['attempted_action_response'] = resp_str

        # 从 env 和 FeatureAgent 提取局面信息
        for s in range(4):
            ag = env.agents[s]
            # 手牌（只有对应座位的 agent 知道自己的手牌）
            hand = list(getattr(ag, 'hand', []))
            # 副露（packs[0] 是相对自己的副露）
            my_packs = []
            if hasattr(ag, 'packs') and len(ag.packs) > 0:
                for pk in ag.packs[0]:  # packs[0] = 自己的副露
                    my_packs.append(list(pk) if isinstance(pk, tuple) else str(pk))
            # 牌河（history[0] 是自己的舍牌）
            discards = []
            if hasattr(ag, 'history') and len(ag.history) > 0:
                discards = list(ag.history[0])

            detail[f'seat_{s}'] = {
                'hand': hand,
                'melds': my_packs,
                'discards': discards,
            }

        # ── 诊断：对比 env 权威状态 vs FeatureAgent 追踪状态 ──
        offender = offender_seat
        # env 权威状态
        detail['env_hand'] = list(env.hands[offender]) if hasattr(env, 'hands') else []
        detail['env_packs'] = [list(pk) if isinstance(pk, tuple) else str(pk) for pk in env.packs[offender]] if hasattr(env, 'packs') and offender < len(env.packs) else []
        detail['env_shownTiles'] = dict(getattr(env, 'shownTiles', {}))
        # FeatureAgent 追踪状态（offender 自己的视角）
        ag = env.agents[offender]
        detail['agent_hand'] = list(getattr(ag, 'hand', []))
        detail['agent_packs'] = [list(pk) if isinstance(pk, tuple) else str(pk) for pk in ag.packs[0]] if hasattr(ag, 'packs') and len(ag.packs) > 0 else []
        detail['agent_shownTiles'] = dict(getattr(ag, 'shownTiles', {}))

        # ── 诊断：检查 offender 选动作时的 action_mask ──
        offender_name = f'player_{offender_seat + 1}'
        if offender_name in last_action_mask:
            mask = last_action_mask[offender_name]
            detail['action_mask_at_hu'] = {
                'hu_value': float(mask[1]),
                'nonzero_count': int((mask > 0).sum()),
                'nonzero_indices': [int(i) for i in np.where(mask > 0)[0][:20]],
            }

        # ── 精确定位：用两边状态分别调用 MahjongFanCalculator ──
        try:
            from MahjongGB import MahjongFanCalculator

            cur_tile = getattr(env, 'curTile', '')
            is_self_drawn = getattr(env, 'state', -1) == 1
            wall_last = getattr(env, 'wallLast', False)
            prev_wind = getattr(env, 'prevalentWind', -1)
            is_about_kong = getattr(env, 'isAboutKong', False)

            # 用 env 权威状态计算
            env_fans_raw = MahjongFanCalculator(
                pack=tuple(env.packs[offender]),
                hand=tuple(env.hands[offender]),
                winTile=cur_tile,
                flowerCount=0,
                isSelfDrawn=is_self_drawn,
                is4thTile=(env.shownTiles[cur_tile] + is_self_drawn) == 4,
                isAboutKong=is_about_kong,
                isWallLast=wall_last,
                seatWind=offender,
                prevalentWind=prev_wind,
                verbose=True,
            )
            env_fan_cnt = sum(fp * cnt for fp, cnt, _, _ in env_fans_raw)
            detail['env_fan_result'] = {'fans': [(fp, cnt, name) for fp, cnt, name, _ in env_fans_raw], 'total': env_fan_cnt}

            # 用 FeatureAgent 状态计算（模拟 _check_mahjong 的输入）
            ag_hand_for_check = list(ag.hand)
            if is_self_drawn and cur_tile in ag_hand_for_check:
                ag_hand_for_check.remove(cur_tile)  # FeatureAgent check is before append
            _shown = ag.shownTiles[cur_tile] + is_self_drawn
            if is_self_drawn:
                _shown += 1  # compensate env's pre-increment

            ag_fans_raw = MahjongFanCalculator(
                pack=tuple(ag.packs[0]),
                hand=tuple(ag_hand_for_check),
                winTile=cur_tile,
                flowerCount=0,
                isSelfDrawn=is_self_drawn,
                is4thTile=_shown == 4,
                isAboutKong=is_about_kong,
                isWallLast=ag.wallLast,
                seatWind=ag.seatWind,
                prevalentWind=ag.prevalentWind,
                verbose=True,
            )
            ag_fan_cnt = sum(fp * cnt for fp, cnt, _, _ in ag_fans_raw)
            detail['agent_fan_result'] = {'fans': [(fp, cnt, name) for fp, cnt, name, _ in ag_fans_raw], 'total': ag_fan_cnt}
        except Exception as e:
            detail['fan_diag_error'] = str(e)

        result['invalid_detail'] = detail

    return result


# ═══════════════════════════════════════════════════════════════════════
# 锦标赛
# ═══════════════════════════════════════════════════════════════════════

def _collect_game_stats(result, seats, stats, model_keys, model_classes, model_paths, invalid_log, deal_id, rot):
    """Collect stats from a single game result into the stats dict."""
    if invalid_log is not None and 'invalid_detail' in result:
        result['invalid_detail']['game_id'] = f'{deal_id}_{rot}'
        invalid_log.append(result['invalid_detail'])

    for seat in range(4):
        mid = seats[seat]
        key = model_keys[mid]
        s = stats[key]
        s['class_name'] = model_classes[mid].__name__
        s['path'] = model_paths[mid]
        s['games'] += 1
        s['total_score'] += result['scores'][seat]

        if result['fan_count'] > 0:
            score = result['scores'][seat]
            rank = sum(1 for other in result['scores'] if other > score) + 1
            s['rank_count'][rank - 1] += 1

        if result['winner'] == seat:
            s['wins'] += 1
            s['total_fan'] += result['fan_count']
            s['fan_history'].append(result['fan_count'])
            if result['is_self_drawn']:
                s['self_draws'] += 1

        if result['deal_in_seat'] == seat:
            s['deal_ins'] += 1

        s['score_history'].append(result['scores'][seat])

        if result['fan_count'] == -1:
            s['invalid_involved'] += 1
            if result['scores'][seat] == -30:
                s['invalid_games'] += 1


def run_tournament(model_paths, model_classes, seat_assignments, num_games,
                   device='cpu', rotate_seats=True, verbose=False,
                   report_interval=50, capture_invalid=False):
    """
    参数:
        model_paths: list[str]，每个 checkpoint 的路径
        model_classes: list[type]，每个模型对应的 PyTorch nn.Module 类
        seat_assignments: list[int]，长度 4，seat_assignments[seat] = model_paths 中的索引
        num_games: int，发牌副数（每副牌打 4 局，所有模型轮换座位，实际总局数 = num_games × 4）
        device: 'cpu' | 'cuda'
        rotate_seats: bool，是否在 deals 之间也轮换基准座位
        verbose: bool，是否打印每步
        report_interval: int，每 N 副牌打印进度

    返回:
        (stats, model_keys, invalid_log)
        stats: dict，按模型 key 汇总的统计
        model_keys: list[str]
        invalid_log: list[dict]，无效局的详细信息（仅 capture_invalid=True 时有内容）
    """
    num_models = len(model_paths)
    print(f'Loading {num_models} model(s)...')
    models = []
    for i, (path, cls) in enumerate(zip(model_paths, model_classes)):
        m = ModelWrapper(cls, path, device=device)
        models.append(m)
        print(f'  Model {i}: {cls.__name__}  ←  {path}  ({m.param_count:,} params)')

    unique_models = len(set(zip(model_paths, model_classes)))
    print(f'Unique models: {unique_models}')
    print(f'Deals: {num_games} (actual games: {num_games * 4})  |  Rotate seats: {rotate_seats}')
    print(f'Seat assignment template: {seat_assignments}')
    print()

    # ── 模型 key 包含路径上下文，避免同名文件冲突 ──
    model_keys = []
    for i, (path, cls) in enumerate(zip(model_paths, model_classes)):
        # 提取路径中的有意义的标识：上级目录 + 文件名
        # 例: log/run_resnet/checkpoint/19.pkl → ResNetModel@run_resnet/19
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        ckpt_name = os.path.splitext(os.path.basename(path))[0]
        key = f'{cls.__name__}@{run_dir}/{ckpt_name}'
        model_keys.append(key)

    stats = defaultdict(lambda: {
        'games': 0,
        'rank_count': [0, 0, 0, 0],
        'wins': 0,
        'self_draws': 0,
        'deal_ins': 0,
        'total_score': 0.0,
        'total_fan': 0.0,
        'score_history': [],
        'fan_history': [],
        'invalid_games': 0,    # 该模型是犯规方（得分 -30）的次数
        'invalid_involved': 0, # 该模型卷入无效局的座位次数
        'class_name': '',
        'path': '',
    })

    invalid_log = []  # 收集无效局的详细信息

    start_time = time.time()

    for deal_id in range(num_games):
        # ── 计算基准座位（每个 deal 的第一局） ──
        if rotate_seats:
            offset = deal_id % 4
            seats_base = [seat_assignments[(s - offset) % 4] for s in range(4)]
        else:
            seats_base = list(seat_assignments)

        # ── 第一局：随机发牌 ──
        env = MahjongGBEnv(config={'agent_clz': RLFeatureAgent})
        result = run_one_game(env, models, seats_base, verbose=verbose,
                              capture_invalid=capture_invalid)
        _collect_game_stats(result, seats_base, stats, model_keys,
                           model_classes, model_paths, invalid_log, deal_id, 0)

        tile_wall = env.originalTileWall
        prev_wind = env.prevalentWind

        # ── 同一副牌，每个模型轮换座位再打 3 局 ──
        for rot in range(1, 4):
            env_r = MahjongGBEnv(config={'agent_clz': RLFeatureAgent})
            env_r.reset(prevalentWind=prev_wind, tileWall=tile_wall)
            seats_rot = [seats_base[(s - rot) % 4] for s in range(4)]
            result = run_one_game(env_r, models, seats_rot, verbose=verbose,
                                  capture_invalid=capture_invalid)
            _collect_game_stats(result, seats_rot, stats, model_keys,
                               model_classes, model_paths, invalid_log, deal_id, rot)

        # ── 进度打印（以 deal 为单位） ──
        if (deal_id + 1) % report_interval == 0 or deal_id == num_games - 1:
            elapsed = time.time() - start_time
            actual_games = (deal_id + 1) * 4
            gps = actual_games / elapsed if elapsed > 0 else 0
            print(f'  [{deal_id + 1}/{num_games} deals, {actual_games} games]  '
                  f'{elapsed:.0f}s  ({gps:.1f} games/s)')

    total_time = time.time() - start_time
    total_games = num_games * 4
    print(f'\nDone. {num_games} deals ({total_games} games) in {total_time:.0f}s '
          f'({total_games / total_time:.1f} games/s)')

    return dict(stats), model_keys, invalid_log


# ═══════════════════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════════════════


def print_report(stats, model_keys, run_dir, invalid_log=None):
    """打印对战报告并保存到文件。"""
    lines = []
    def emit(s=''):
        lines.append(s)
        print(s)

    emit('=' * 72)
    emit('  BATTLE REPORT')
    emit('=' * 72)

    # ── 模型信息 ──
    emit('\nModels:')
    for i, key in enumerate(model_keys):
        s = stats[key]
        emit(f'  [{i}] {s["class_name"]}')
        emit(f'      Path: {s["path"]}')
        emit(f'      Games played: {s["games"]}')

    # ── 汇总表格 ──
    emit(f'\n{"Model":<30} {"Games":>6} {"Win%":>7} {"SelfDr%":>8} '
         f'{"DealIn%":>8} {"AvgScore":>9} {"AvgFan":>7} {"AvgRank":>8}')
    emit('-' * 87)

    # 用短标签做表头
    def label_from_key(k):
        s = stats[k]
        d = os.path.basename(os.path.dirname(os.path.dirname(s['path'])))
        f = os.path.splitext(os.path.basename(s['path']))[0]
        return f'{s["class_name"]}@{d}/{f}'

    labels = [label_from_key(k) for k in model_keys]

    for key, label in zip(model_keys, labels):
        s = stats[key]
        g = s['games']
        if g == 0:
            continue
        ranked = sum(s['rank_count'])  # 有明确名次的局数
        win_pct = s['wins'] / g * 100
        sd_pct = s['self_draws'] / g * 100
        di_pct = s['deal_ins'] / g * 100
        avg_score = s['total_score'] / g
        avg_fan = s['total_fan'] / s['wins'] if s['wins'] > 0 else 0
        avg_rank = sum((i + 1) * c for i, c in enumerate(s['rank_count'])) / ranked if ranked > 0 else 0

        emit(f'{label:<30} {g:>6} {win_pct:>6.1f}% {sd_pct:>7.1f}% '
             f'{di_pct:>7.1f}% {avg_score:>9.2f} {avg_fan:>7.1f} {avg_rank:>8.3f}')

    # ── 排名分布 ──
    emit('\nRank distribution:')
    emit(f'{"Model":<30} {"1st":>8} {"2nd":>8} {"3rd":>8} {"4th":>8} {"Unranked":>10}')
    emit('-' * 78)
    for key, label in zip(model_keys, labels):
        s = stats[key]
        g = s['games']
        if g == 0:
            continue
        ranked = sum(s['rank_count'])
        unranked = g - ranked
        ranks = [f'{c/g*100:5.1f}%' for c in s['rank_count']]
        emit(f'{label:<30} {ranks[0]:>8} {ranks[1]:>8} {ranks[2]:>8} {ranks[3]:>8} {unranked/g*100:>9.1f}%')

    # ── 无效动作 ──
    emit('\nInvalid action rate (model as offender / score = -30):')
    for key, label in zip(model_keys, labels):
        s = stats[key]
        g = s['games']
        if g > 0:
            emit(f'  {label}: {s["invalid_games"]}/{g} ({s["invalid_games"]/g*100:.1f}%)  '
                 f'(involved in {s["invalid_involved"]}/{g} invalid games)')

    emit(f'\nReport saved to: {run_dir}')

    # ── 保存到文件 ──
    report_file = os.path.join(run_dir, 'battle_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # ── 保存统计 JSON ──
    json_stats = {}
    for key, label in zip(model_keys, labels):
        s = stats[key]
        g = s['games']
        ranked = sum(s['rank_count'])
        json_stats[label] = {
            'class': s['class_name'],
            'path': s['path'],
            'games': g,
            'ranked_games': ranked,
            'unranked_games': g - ranked,
            'wins': s['wins'],
            'win_pct': round(s['wins'] / g * 100, 2) if g > 0 else 0,
            'self_draws': s['self_draws'],
            'deal_ins': s['deal_ins'],
            'avg_score': round(s['total_score'] / g, 2) if g > 0 else 0,
            'avg_fan': round(s['total_fan'] / s['wins'], 1) if s['wins'] > 0 else 0,
            'avg_rank': round(sum((i + 1) * c for i, c in enumerate(s['rank_count'])) / ranked, 3) if ranked > 0 else 0,
            'rank_distribution': [s['rank_count'][i] for i in range(4)],
            'rank_pct': [round(s['rank_count'][i] / g * 100, 1) for i in range(4)] if g > 0 else [],
            'unranked_pct': round((g - ranked) / g * 100, 1) if g > 0 else 0,
            'invalid_as_offender': s['invalid_games'],
            'invalid_involved': s['invalid_involved'],
            'score_history': s['score_history'],
            'fan_history': s['fan_history'],
        }
    json_file = os.path.join(run_dir, 'battle_stats.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)

    # ── 保存无效动作详细日志 ──
    if invalid_log:
        invalid_file = os.path.join(run_dir, 'invalid_log.json')
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_log, f, indent=2, ensure_ascii=False)
        print(f'Invalid action log saved: {invalid_file} ({len(invalid_log)} games)')

    # ── 保存 CSV ──
    csv_file = os.path.join(run_dir, 'battle_summary.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'class', 'games', 'win_pct', 'self_draw_pct', 'deal_in_pct',
                         'avg_score', 'avg_fan', 'avg_rank',
                         'rank1_pct', 'rank2_pct', 'rank3_pct', 'rank4_pct', 'unranked_pct'])
        for key, label in zip(model_keys, labels):
            s = stats[key]
            g = s['games']
            if g == 0:
                continue
            ranked = sum(s['rank_count'])
            writer.writerow([
                label, s['class_name'], g,
                round(s['wins'] / g * 100, 2),
                round(s['self_draws'] / g * 100, 2),
                round(s['deal_ins'] / g * 100, 2),
                round(s['total_score'] / g, 2),
                round(s['total_fan'] / s['wins'], 1) if s['wins'] > 0 else 0,
                round(sum((i + 1) * c for i, c in enumerate(s['rank_count'])) / ranked, 3) if ranked > 0 else 0,
                round(s['rank_count'][0] / g * 100, 1),
                round(s['rank_count'][1] / g * 100, 1),
                round(s['rank_count'][2] / g * 100, 1),
                round(s['rank_count'][3] / g * 100, 1),
                round((g - ranked) / g * 100, 1),
            ])

    # ── 图表 ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

        # 1. 排名分布
        ax = axes[0]
        x = np.arange(len(labels))
        width = 0.15
        rank_labels = ['1st', '2nd', '3rd', '4th', 'Unranked']
        rank_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
        for ri in range(5):
            vals = []
            for key in model_keys:
                g = stats[key]['games']
                if ri < 4:
                    vals.append(stats[key]['rank_count'][ri] / g * 100 if g > 0 else 0)
                else:
                    ranked = sum(stats[key]['rank_count'])
                    vals.append((g - ranked) / g * 100 if g > 0 else 0)
            ax.bar(x + (ri - 2) * width, vals, width, label=rank_labels[ri], color=rank_colors[ri])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%')
        ax.set_title('Rank Distribution')
        ax.legend(fontsize=8)

        # 2. 得分箱线图
        ax = axes[1]
        score_data = [stats[k]['score_history'] for k in model_keys]
        bp = ax.boxplot(score_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title('Score Distribution per Game')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', labelsize=8)

        # 3. 累计得分
        ax = axes[2]
        for i, key in enumerate(model_keys):
            scores = stats[key]['score_history']
            cumulative = np.cumsum(scores)
            ax.plot(range(1, len(cumulative) + 1), cumulative, label=labels[i],
                    color=colors[i % len(colors)], linewidth=1.5)
        ax.set_xlabel('Game')
        ax.set_ylabel('Cumulative Score')
        ax.set_title('Cumulative Score Over Games')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(run_dir, 'battle_charts.png')
        plt.savefig(plot_file, dpi=200)
        plt.close()
        print(f'Charts saved: {plot_file}')
    except ImportError:
        print('matplotlib not installed, skipping charts.')
    except Exception as ex:
        print(f'Chart generation failed: {ex}')

    return lines


# ═══════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='SL 麻将模型对战评估 — 支持不同网络结构的模型混战',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 2模型对战（默认 model.CNNModel），交叉坐位
  python battle.py --models a.pkl b.pkl --games 500

  # ★ 不同网络结构对战：baseline用浅层CNN，新版用ResNet
  python battle.py \\
      --models log/baseline/19.pkl log/resnet/19.pkl \\
      --model-classes model.CNNModel model.ResNetModel \\
      --games 500

  # 4个不同结构的模型各坐1位
  python battle.py \\
      --models m0.pkl m1.pkl m2.pkl m3.pkl \\
      --model-classes model.CNNModel model.ResNetModel model.ViTModel my_pkg.MyModel \\
      --games 200

  # 自对弈（默认 model.CNNModel）
  python battle.py --models baseline.pkl --selfplay --games 500

  # 自定义模型类 + 自对弈
  python battle.py --models resnet.pkl --model-classes model.ResNetModel --selfplay --games 200
        """
    )
    p.add_argument('--models', nargs='+', required=True,
                   help='模型 checkpoint 路径，按顺序编号 0,1,2,...')
    p.add_argument('--model-classes', nargs='+', default=None,
                   help='每个模型对应的类 (格式: module.ClassName)。'
                        '数量须与 --models 一致。'
                        '默认全部为 model.CNNModel。'
                        '示例: model.CNNModel model.ResNetModel')
    p.add_argument('--seats', nargs=4, type=int, default=None,
                   help='4个座位分别使用哪个模型索引 (默认: 2模型时交叉坐, 4模型时各坐1位)')
    p.add_argument('--games', type=int, default=200,
                   help='总局数 (默认: 200)')
    p.add_argument('--device', default='cpu',
                   help='推理设备 (默认: cpu)')
    p.add_argument('--no-rotate', action='store_true',
                   help='禁止座位轮换')
    p.add_argument('--selfplay', action='store_true',
                   help='自对弈模式 (4家都使用第一个模型)')
    p.add_argument('--verbose', action='store_true',
                   help='打印每步动作 (调试用)')
    p.add_argument('--report-interval', type=int, default=50,
                   help='每N局打印进度 (默认: 50)')
    p.add_argument('--output', default=None,
                   help='输出目录 (默认: battle_results/run_<timestamp>)')
    p.add_argument('--invalid-log', action='store_true',
                   help='记录每局无效动作的详细局面（手牌、副露、牌河、尝试的动作），'
                        '保存到 invalid_log.json')
    return p.parse_args()


def main():
    args = parse_args()

    # ── 模型路径 ──
    if args.selfplay:
        model_paths = [args.models[0]]
    else:
        model_paths = list(args.models)

    # ── 模型类（动态导入） ──
    if args.model_classes:
        if len(args.model_classes) != len(model_paths):
            p.error(
                f'--model-classes 数量 ({len(args.model_classes)}) '
                f'与 --models 数量 ({len(model_paths)}) 不一致。'
            )
        model_classes = []
        for spec in args.model_classes:
            cls = import_model_class(spec)
            model_classes.append(cls)
            print(f'Imported model class: {spec} → {cls.__name__}')
    else:
        # 默认全部使用 model.CNNModel
        from model import CNNModel as DefaultModel
        model_classes = [DefaultModel] * len(model_paths)

    # ── 座位分配 ──
    if args.seats:
        seat_assignments = list(args.seats)
        for s in seat_assignments:
            if s < 0 or s >= len(model_paths):
                raise ValueError(f'Seat index {s} out of range for {len(model_paths)} models')
    else:
        if len(model_paths) == 1:
            seat_assignments = [0, 0, 0, 0]
        elif len(model_paths) == 2:
            seat_assignments = [0, 1, 0, 1]
        elif len(model_paths) == 4:
            seat_assignments = [0, 1, 2, 3]
        else:
            raise ValueError(
                f'无法自动为 {len(model_paths)} 个模型分配4个座位。'
                f'请使用 --seats 手动指定或提供 1/2/4 个模型。'
            )

    # ── 输出目录 ──
    if args.output:
        run_dir = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(_SL_DIR, 'battle_results', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # ── 保存配置 ──
    config = {
        'models': model_paths,
        'model_classes': [cls.__name__ for cls in model_classes],
        'seat_assignments': seat_assignments,
        'num_games': args.games,
        'rotate_seats': not args.no_rotate,
        'selfplay': args.selfplay,
        'device': args.device,
    }
    with open(os.path.join(run_dir, 'battle_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'Battle config:')
    for k, v in config.items():
        print(f'  {k}: {v}')
    print(f'  output: {run_dir}\n')

    stats, model_keys, invalid_log = run_tournament(
        model_paths=model_paths,
        model_classes=model_classes,
        seat_assignments=seat_assignments,
        num_games=args.games,
        device=args.device,
        rotate_seats=not args.no_rotate,
        verbose=args.verbose,
        report_interval=args.report_interval,
        capture_invalid=args.invalid_log,
    )

    print_report(stats, model_keys, run_dir, invalid_log)


if __name__ == '__main__':
    main()
