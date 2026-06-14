# dataset.py — 麻将数据集（延迟加载 + 数据增强）
#
# ── 原有代码 ──
# MahjongGBDataset 继承 PyTorch Dataset，为 DataLoader 提供数据。
# __init__ 时从 data/count.json 读取每局样本数，构建前缀和数组，
# 然后全量预加载所有 .npz 文件到 self.cache 字典中。
# __getitem__ 通过二分查找定位样本所属对局，从缓存直接取。
#
# ── 新增 ──
# 1. 延迟加载（LRU 文件缓存）
#    145 维特征时数据量大幅膨胀（每样本 ~5.5KB），全量加载会 OOM。
#    改为按需加载，内存中只保留最近访问的 cache_size 个 match 的数据。
#    实现：OrderedDict 做 LRU — 访问时 move_to_end，满时 popitem(last=False)。
#
# 2. 数据增强（IJCAI 第4名方案，Slide 9）
#    每次 __getitem__ 以 50% 概率做两种变换的叠加：
#    (a) 花色互换：万/条/筒 3 色行排列（row 0,1,2），6 种可能
#    (b) 数字反转：整个数轴 1-9 反转（1↔9, 2↔8, 3↔7, 4↔6, 5不动），2 种可能
#    合计 12 倍增强。变换时同步更新 obs、action_mask、action_label，
#    保证 "变换后的局面 + 变换后的动作" 完全等价于原数据。
#
#    关键：数字反转必须四对同时进行，不能独立翻转。
#    如果只翻 (1,9) 而不翻 (2,8)，顺子 [1,2,3] 会变成 [9,2,3] —
#    不再连续，吃牌动作映射会出错。四对全翻时 [1,2,3]→[9,8,7] 仍是顺子。
#
#    注意：花色互换和数字反转只适用于 suited tiles（万条筒），
#    风牌（F1-F4）和箭牌（J1-J3）不参与变换。
#    论文指出：推不倒、绿一色等特殊番型在这种增强下会失真，
#    但这些番型出现频率极低，对训练影响可忽略。

from torch.utils.data import Dataset
from collections import OrderedDict
from bisect import bisect_right
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 数据增强 — 静态辅助函数
#
# 数字反转原理（保证吃牌连续性）：
#   1-9 的数轴整体反转：1↔9, 2↔8, 3↔7, 4↔6, 5↔5
#   必须四对同时翻转，不能独立翻转。
#   独立翻转会破坏顺子：只翻 (1,9) 时 [1,2,3]→[9,2,3] 不连续
#   四对全翻时 [1,2,3]→[9,8,7]→排序 [7,8,9] 仍是顺子 ✓
# ═══════════════════════════════════════════════════════════════════════


def _build_action_map(suit_perm, do_mirror):
    """
    构建 action 索引的变换表：action_map[old_action] = new_action。

    参数：
        suit_perm: 长度为 3 的排列，suit_perm[i] = 原花色 i 变换到哪个花色行
        do_mirror: bool，是否做数字反转（1-9 整条数轴反转）

    Action 空间布局（来自 feature.py OFFSET_ACT）：
        0:    Pass（不变）
        1:    Hu（不变）
        2-35:  Play = 2 + tile_idx
        36-98: Chi  = 36 + suit*21 + (center-2)*3 + position
        99-132:  Peng   = 99 + tile_idx
        133-166: Gang   = 133 + tile_idx（明杠）
        167-200: AnGang = 167 + tile_idx（暗杠）
        201-234: BuGang = 201 + tile_idx（补杠）
    """
    action_map = np.arange(235, dtype=int)

    # ── 1. tile 映射 ──
    # TILE_LIST: W1..W9(0-8), T1..T9(9-17), B1..B9(18-26),
    #            F1..F4(27-30), J1..J3(31-33)
    tile_map = np.arange(34, dtype=int)
    for old_suit in range(3):
        new_suit = suit_perm[old_suit]
        for col in range(9):
            old_idx = old_suit * 9 + col
            new_col = (8 - col) if do_mirror else col  # 数字反转：整条数轴翻转
            new_idx = new_suit * 9 + new_col
            tile_map[old_idx] = new_idx

    # ── 2. Play / Peng / Gang / AnGang / BuGang — 只受 tile 映射影响 ──
    for base in [2, 99, 133, 167, 201]:
        for old_tile in range(34):
            action_map[base + old_tile] = base + tile_map[old_tile]

    # ── 3. Chi — 花色 + 数字反转都影响 ──
    # Chi action = 36 + suit*21 + (center-2)*3 + position
    # 数字反转后：old_tiles (c-1,c,c+1) 各列翻转为 (8-col)
    #   例: center=3(列2), pos=1 → tiles 列[1,2,3]→[7,6,5] 排序[5,6,7]
    #   → new_center=7, new_pos=1 (offered 列6 在排序中的位置)
    for old_suit in range(3):
        for old_center in range(2, 9):
            for old_pos in range(3):
                old_action = 36 + old_suit * 21 + (old_center - 2) * 3 + old_pos
                if do_mirror:
                    nums = [old_center - 2, old_center - 1, old_center]         # 三张牌 0-based 列
                    offered = nums[old_pos]
                    new_nums = [8 - n for n in nums]                            # 数字反转
                    new_offered = 8 - offered
                    sorted_new = sorted(new_nums)
                    new_center = sorted_new[1] + 1                              # 1-based
                    new_pos = sorted_new.index(new_offered)
                else:
                    new_center = old_center
                    new_pos = old_pos
                new_suit = suit_perm[old_suit]
                action_map[old_action] = 36 + new_suit * 21 + (new_center - 2) * 3 + new_pos

    return action_map


def _augment_obs(obs, suit_perm, do_mirror):
    """
    对 observation tensor 做数据增强。

    obs 形状: (C, 4, 9) — C 个通道 × 4 行（W/T/B/FJ）× 9 列（数字 1-9）

    变换：
        (a) 花色互换：排列 rows 0,1,2，row 3（风箭）不动
        (b) 数字反转：rows 0-2 的列顺序整体反转（9→1），row 3 不动
    """
    obs = obs.copy()

    # (a) 花色行排列
    obs = obs[:, list(suit_perm) + [3], :]

    # (b) 数字反转：0-2行整体列反转
    if do_mirror:
        obs[:, :3, :] = obs[:, :3, ::-1]

    return obs


class MahjongGBDataset(Dataset):
    """
    国标麻将监督学习数据集。

    参数：
        begin:  训练/验证集起始比例 (0.0 ~ 1.0)
        end:    训练/验证集结束比例 (0.0 ~ 1.0)
                例: (0, 0.9)=训练集, (0.9, 1)=验证集
        augment: 是否启用数据增强（训练集 True，验证集 False）
        cache_size: LRU 文件缓存大小（默认 16 个 match）
    """

    def __init__(self, begin=0, end=1, augment=False, cache_size=0):
        """
        参数：
            cache_size: LRU 文件缓存大小。
                        0（默认）= 全量预加载到内存（快，6 维用这个）
                        >0 = 延迟加载，缓存最近 N 个 match（省内存，145 维用这个）
        """
        import json

        # ── 读取每局样本数 ──
        with open('data/count.json') as f:
            all_match_samples = json.load(f)

        self.total_matches = len(all_match_samples)
        self.total_samples = sum(all_match_samples)

        # ── 按 begin/end 比例切片 ──
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = all_match_samples[self.begin:self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        self.cache_size = cache_size

        # ── 保留原始每局样本数（前缀和之前） ──
        self._match_counts = list(self.match_samples)

        # ── 构建前缀和数组 ──
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a

        # ── 文件分组 shuffle 索引（cache_size>0 时使用，避免随机IO） ──
        self._sample_index = None
        if cache_size > 0:
            self._build_shuffled_index()

        # ── 加载数据 ──
        if cache_size > 0:
            # 延迟加载模式（145 维用）：LRU 文件缓存
            self._file_cache = OrderedDict()
            self._full_cache = None
        else:
            # 全量预加载模式（6 维用）：一次性全读入内存
            self._file_cache = None
            self._full_cache = {'obs': [], 'mask': [], 'act': []}
            for i in range(self.matches):
                if i % 128 == 0:
                    print('loading', i)
                d = np.load('data/%d.npz' % (i + self.begin))
                for k in d:
                    self._full_cache[k].append(d[k])

        # ── 预计算 12 种增强的 action_map + inv_map ──
        # 增强只有 12 种组合（6 suit_perm × 2 do_mirror），
        # 提前算好避免每次 __getitem__ 重复计算 + argsort
        if augment:
            from itertools import permutations
            self._aug_pool = []
            for suit_perm in permutations([0, 1, 2]):
                for do_mirror in (False, True):
                    am = _build_action_map(np.array(suit_perm), do_mirror)
                    inv = np.argsort(am)
                    suit_perm_arr = np.array(suit_perm)
                    self._aug_pool.append((suit_perm_arr, do_mirror, am, inv))

    def _build_shuffled_index(self):
        """构建按文件分组的 shuffle 索引，同局样本连续存放，让每个文件只读一次。"""
        import random as _random
        match_order = list(range(self.matches))
        _random.shuffle(match_order)
        self._sample_index = []
        for m in match_order:
            n = self._match_counts[m]
            for s in range(n):
                self._sample_index.append((m, s))

    def reshuffle(self):
        """每 epoch 调用一次，重新随机排列对局顺序。"""
        if self._sample_index is not None:
            self._build_shuffled_index()

    def _load_match(self, match_id):
        """延迟加载：从 LRU 缓存或磁盘获取 match 数据。"""
        if match_id in self._file_cache:
            self._file_cache.move_to_end(match_id)
            return self._file_cache[match_id]

        while len(self._file_cache) >= self.cache_size:
            self._file_cache.popitem(last=False)

        d = np.load('data/%d.npz' % match_id)
        data = {'obs': d['obs'], 'mask': d['mask'], 'act': d['act']}
        self._file_cache[match_id] = data
        return data

    def _get_match_data(self, match_id):
        """根据加载模式获取 match 数据。"""
        if self._full_cache is not None:
            # 全量预加载：直接从 list 取
            return self._full_cache
        else:
            # 延迟加载：走 LRU
            return self._load_match(match_id + self.begin)

    def __len__(self):
        if self._sample_index is not None:
            return len(self._sample_index)
        return self.samples

    def __getitem__(self, index):
        if self._sample_index is not None:
            match_id, sample_id = self._sample_index[index]
        else:
            match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
            sample_id = index - self.match_samples[match_id]

        # 全量预加载模式 vs 延迟加载模式
        if self._full_cache is not None:
            obs = self._full_cache['obs'][match_id][sample_id]
            mask = self._full_cache['mask'][match_id][sample_id]
            act = self._full_cache['act'][match_id][sample_id]
        else:
            match_data = self._load_match(match_id + self.begin)
            obs = match_data['obs'][sample_id]
            mask = match_data['mask'][sample_id]
            act = match_data['act'][sample_id]

        # ── 数据增强 ──
        if self.augment and np.random.random() < 0.5:
            # 从预计算的 12 种增强中随机选一种（避免每次现场构建 action_map）
            suit_perm, do_mirror, action_map, inv_map = self._aug_pool[
                np.random.randint(len(self._aug_pool))]

            obs = _augment_obs(obs, suit_perm, do_mirror)
            mask = mask[inv_map]
            act = action_map[act]

        return obs, mask, act
