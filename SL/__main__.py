# Agent part
from feature import FeatureAgent, FeatureAgent145

# Model part
from model import CNNModel
from model import ResNetModel
# Botzone interaction
import numpy as np
import torch

def obs2response(model, obs):
    logits = model({'is_training': False, 'obs': {'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)), 'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))}})
    action = logits.detach().numpy().flatten().argmax()
    response = agent.action2response(action)
    return response

def _detect_from_checkpoint(state_dict):
    """从 checkpoint 自动检测 in_channels 和模型类型。"""
    in_channels = 6
    is_resnet = False
    for k, v in state_dict.items():
        if len(v.shape) == 4 and v.shape[0] >= 64:
            in_channels = v.shape[1]
            break
    for k in state_dict.keys():
        if k.startswith('stem.'):
            is_resnet = True; break
        if k.startswith('_tower.'):
            break
    return in_channels, is_resnet

import sys

if __name__ == '__main__':
    data_dir = 'data/mahjong2.pkl'
    state = torch.load(data_dir, map_location=torch.device('cpu'))
    in_channels, is_resnet = _detect_from_checkpoint(state)

    if is_resnet:
        model = ResNetModel(in_channels=in_channels)
    else:
        model = CNNModel(in_channels=in_channels)
    model.load_state_dict(state)

    # 根据通道数选择匹配的 FeatureAgent（6维 vs 145维）
    AgentClass = FeatureAgent145 if in_channels >= 100 else FeatureAgent

    angang = None
    zimo = False
    input()
    while True:
        request = input()
        while not request.strip(): request = input()
        t = request.split()
        if t[0] == '0':
            seatWind = int(t[1])
            agent = AgentClass(seatWind)
            agent.request2obs('Wind %s' % t[2])
            print('PASS')
        elif t[0] == '1':
            agent.request2obs(' '.join(['Deal', *t[5:]]))
            print('PASS')
        elif t[0] == '2':
            obs = agent.request2obs('Draw %s' % t[1])
            response = obs2response(model, obs)
            t = response.split()
            if t[0] == 'Hu':
                print('HU')
            elif t[0] == 'Play':
                print('PLAY %s' % t[1])
            elif t[0] == 'Gang':
                print('GANG %s' % t[1])
                angang = t[1]
            elif t[0] == 'BuGang':
                print('BUGANG %s' % t[1])
        elif t[0] == '3':
            p = int(t[1])
            if t[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')
            elif t[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif t[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')
            else:
                zimo = False
                if t[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    t = response.split()
                    if t[0] == 'Hu':
                        print('HU')
                    elif t[0] == 'Pass':
                        print('PASS')
                    elif t[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif t[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d '% seatWind + response)
                        response2 = obs2response(model, obs)
                        print(' '.join([t[0].upper(), *t[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + response)
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()
#5
#0 1 0
#1 0 0 0 0 T5 W5 T6 F1 W7 F4 B2 B5 B9 B3 W6 W5 W7
#3 0 DRAW
#3 0 PLAY F4
#2 B4


#
#
