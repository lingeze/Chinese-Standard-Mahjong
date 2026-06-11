from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
import torch.nn as nn
#from model import CNNModel
import torch.nn.functional as F
import torch
import os
import csv
import time
import json
import sys
import numpy as np
from datetime import datetime

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1) :
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.bn2 = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return y

class BottleNeck(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1) :
        super().__init__()
        mid_channels = input_channels
        if input_channels / 4 != 0:
            mid_channels = int(mid_channels / 4)
        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=1, padding=0, stride=strides)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(mid_channels, output_channels, kernel_size=1, padding=0, stride=strides)
        if use_1x1conv:
            self.conv4 = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.bn3 = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        if self.conv4:
            x = self.conv4(x)
        y = y + x
        return y

# ── Action type definitions (consistent with feature.py OFFSET_ACT) ──
ACT_TYPE_NAMES = ['Pass', 'Hu', 'Play', 'Chi', 'Peng', 'Gang', 'AnGang', 'BuGang']
ACT_TYPE_OFFSETS = [0, 1, 2, 36, 99, 133, 167, 201]
ACT_TYPE_COUNTS = [1, 1, 34, 63, 34, 34, 34, 34]

def action_to_type(action_idx):
    """Map an action index (0..234) to its type index (0..7)."""
    for i in range(len(ACT_TYPE_OFFSETS) - 1, -1, -1):
        if action_idx >= ACT_TYPE_OFFSETS[i]:
            return i
    return 0


# ═══════════════════════════════════════════════════════════════════════
# 超参数收集 — 训练结束后保存完整配置，便于事后回溯
# ═══════════════════════════════════════════════════════════════════════

def save_hyperparams(run_dir, model, optimizer, scheduler, train_dataset,
                     val_dataset, batch_size, num_epochs, split_ratio,
                     loss_fn_name='CrossEntropyLoss', device='cuda'):
    """
    收集所有超参数和训练配置，写入 run_dir/hyperparameters.json。
    """
    from feature import FeatureAgent

    hp = {}

    # ── 基本信息 ──
    hp['timestamp'] = os.path.basename(run_dir).replace('run_', '')
    hp['environment'] = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
    }

    # ── 数据集 ──
    hp['dataset'] = {
        'class': type(train_dataset).__name__,
        'total_matches': train_dataset.total_matches,
        'total_samples': train_dataset.total_samples,
        'split_ratio': split_ratio,
        'train_matches': train_dataset.matches,
        'train_samples': train_dataset.samples,
        'val_matches': val_dataset.matches,
        'val_samples': val_dataset.samples,
        'augment_train': getattr(train_dataset, 'augment', None),
        'augment_val': getattr(val_dataset, 'augment', None),
        'batch_size': batch_size,
        'loader_shuffle_train': True,
        'loader_shuffle_val': False,
    }

    # ── 特征 ──
    hp['feature'] = {
        'agent_class': FeatureAgent.__name__,
        'OBS_SIZE': FeatureAgent.OBS_SIZE,
        'obs_shape': f'{FeatureAgent.OBS_SIZE}×4×9',
        'ACT_SIZE': FeatureAgent.ACT_SIZE,
        'obs_channels': dict(FeatureAgent.OFFSET_OBS),
        'action_space_breakdown': {
            'Pass': 1,
            'Hu': 1,
            'Play': 34,
            'Chi': 63,
            'Peng': 34,
            'Gang': 34,
            'AnGang': 34,
            'BuGang': 34,
            'total': 235,
        },
    }

    # ── 模型 ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hp['model'] = {
        'class': type(model).__name__,
        'module': type(model).__module__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'full_architecture': str(model),
    }

    # ── 优化器 ──
    opt_cfg = optimizer.param_groups[0]
    hp['optimizer'] = {
        'type': type(optimizer).__name__,
        'lr': opt_cfg['lr'],
        'betas': opt_cfg.get('betas', None),
        'eps': opt_cfg.get('eps', None),
        'weight_decay': opt_cfg.get('weight_decay', 0),
    }

    # ── 学习率调度器 ──
    hp['scheduler'] = {
        'type': type(scheduler).__name__,
    }
    if hasattr(scheduler, 'step_size'):
        hp['scheduler']['step_size'] = scheduler.step_size
    if hasattr(scheduler, 'gamma'):
        hp['scheduler']['gamma'] = scheduler.gamma

    # ── 训练 ──
    hp['training'] = {
        'num_epochs': num_epochs,
        'loss_function': loss_fn_name,
        'label_smoothing': 0.0,
    }

    # ── 写入文件 ──
    hp_file = os.path.join(run_dir, 'hyperparameters.json')
    with open(hp_file, 'w', encoding='utf-8') as f:
        json.dump(hp, f, indent=2, ensure_ascii=False)
    print(f'Hyperparameters saved: {hp_file}')

class MyResNet(nn.Module):

    def __init__(self) :
        super().__init__()
        self.bl1 = nn.Sequential(nn.Conv1d(141, 256, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm1d(256), nn.ReLU()
                                 )
        self.bl2 = nn.Sequential(*self.res_layer_maker(256, 256, 5, first_block=True))
        self.bl3 = nn.Sequential(*self.res_layer_maker(256, 512, 5))
        self.bl4 = nn.Sequential(*self.res_layer_maker(512, 1024, 5))
        self.bottleneck = BottleNeck(1024,1024)
        self.bl5 = nn.Sequential(*self.res_layer_maker(1024,1024,5))
        #self.pool1
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024*5, 44)

    def forward(self, x):
        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bottleneck(x)
        x = self.bl5(x)
        x = self.linear(self.flatten(x))
        return F.log_softmax(x, dim=1)

    @staticmethod
    def res_layer_maker(input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResBlock(input_channels, output_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResBlock(output_channels, output_channels))
        return blk

if __name__ == '__main__':
    logdir = 'log/'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(logdir, 'run_' + timestamp)
    os.makedirs(os.path.join(run_dir, 'checkpoint'), exist_ok=True)

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, augment=True)
    validateDataset = MahjongGBDataset(splitRatio, 1, augment=False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)

    # Load model
    from model import ResNetModel
    model = ResNetModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)

    # Save model architecture
    arch_file = os.path.join(run_dir, 'model_architecture.txt')
    with open(arch_file, 'w') as f:
        f.write(str(model))
        f.write('\n\n=== Parameter Count ===\n')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f'Total parameters: {total_params:,}\n')
        f.write(f'Trainable parameters: {trainable_params:,}\n')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Save all hyperparameters for future reference ──
    save_hyperparams(
        run_dir=run_dir, model=model, optimizer=optimizer,
        scheduler=scheduler, train_dataset=trainDataset,
        val_dataset=validateDataset, batch_size=batchSize,
        num_epochs=20, split_ratio=splitRatio,
        loss_fn_name='CrossEntropyLoss', device='cuda',
    )

    # Log file — overall metrics
    log_file = os.path.join(run_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss_avg', 'train_loss_last', 'val_acc_top1', 'val_acc_top3', 'val_loss', 'lr', 'epoch_time_s'])

    # Log file — per-action-type accuracy per epoch
    type_log_file = os.path.join(run_dir, 'per_type_accuracy.csv')
    with open(type_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch'] + ACT_TYPE_NAMES)

    total_batches = len(trainDataset) // batchSize + 1
    last_confusion = None

    # Train and validate
    for e in range(20):
        print('Epoch', e, 'LR:', optimizer.param_groups[0]['lr'])
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        last_loss = 0.0

        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            train_loss_sum += loss.item()
            train_batches += 1
            last_loss = loss.item()
            if i % 128 == 0:
                print('  Iteration %d/%d'%(i, total_batches), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss_avg = train_loss_sum / train_batches
        torch.save(model.state_dict(), run_dir + '/checkpoint/%d.pkl' % e)

        # Validation
        print('Run validation:')
        model.eval()
        val_correct_top1 = 0
        val_correct_top3 = 0
        val_loss_sum = 0.0
        val_batches = 0

        # Per-action-type tracking
        num_types = len(ACT_TYPE_NAMES)
        type_correct = [0] * num_types
        type_total = [0] * num_types
        confusion = [[0] * num_types for _ in range(num_types)]  # rows=true, cols=pred

        with torch.no_grad():
            for d in vloader:
                input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
                logits = model(input_dict)
                loss = F.cross_entropy(logits, d[2].long().cuda())
                val_loss_sum += loss.item()
                val_batches += 1

                pred = logits.argmax(dim=1)
                targets = d[2].cuda()
                val_correct_top1 += torch.eq(pred, targets).sum().item()

                # Top-3 accuracy
                top3 = logits.topk(3, dim=1).indices
                val_correct_top3 += torch.eq(top3, targets.unsqueeze(1)).any(dim=1).sum().item()

                # Per-type accuracy & confusion matrix
                for t, p in zip(targets.cpu().tolist(), pred.cpu().tolist()):
                    t_type = action_to_type(t)
                    p_type = action_to_type(p)
                    type_total[t_type] += 1
                    if t == p:
                        type_correct[t_type] += 1
                    confusion[t_type][p_type] += 1

        acc_top1 = val_correct_top1 / len(validateDataset)
        acc_top3 = val_correct_top3 / len(validateDataset)
        val_loss_avg = val_loss_sum / val_batches
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print('  Val acc (top-1): %.4f  Val acc (top-3): %.4f  Val loss: %.6f  Time: %.1fs' % (acc_top1, acc_top3, val_loss_avg, epoch_time))

        # ── Per-action-type accuracy ──
        print('  Per-type accuracy:')
        type_acc_strs = []
        for i in range(num_types):
            if type_total[i] > 0:
                acc = type_correct[i] / type_total[i]
                type_acc_strs.append('%s:%.4f(%d)' % (ACT_TYPE_NAMES[i], acc, type_total[i]))
            else:
                type_acc_strs.append('%s:N/A(0)' % ACT_TYPE_NAMES[i])
        print('    ' + '  '.join(type_acc_strs))
        last_confusion = confusion  # retain for saving after training

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e, '%.6f' % train_loss_avg, '%.6f' % last_loss, '%.4f' % acc_top1, '%.4f' % acc_top3, '%.6f' % val_loss_avg, current_lr, '%.1f' % epoch_time])

        with open(type_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [e] + ['%.4f' % (type_correct[i] / type_total[i]) if type_total[i] > 0 else 'N/A' for i in range(num_types)]
            writer.writerow(row)

    # Plot training curves
    print('\nGenerating training curves...')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs, train_loss, val_loss, top1, top3 = [], [], [], [], []
        with open(log_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss_avg']))
                val_loss.append(float(row['val_loss']))
                top1.append(float(row['val_acc_top1']))
                top3.append(float(row['val_acc_top3']))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_loss, label='Train Loss')
        axes[0].plot(epochs, val_loss, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, top1, label='Top-1 Acc', marker='o')
        axes[1].plot(epochs, top3, label='Top-3 Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_file = os.path.join(run_dir, 'training_curves.png')
        plt.savefig(plot_file, dpi=200)
        plt.close()
        print('Plot saved:', plot_file)

        # ── Confusion matrix heatmap ──
        if last_confusion is not None:
            fig, ax = plt.subplots(figsize=(8, 7))
            cm = np.array(last_confusion, dtype=np.float64)
            # Normalize each row to show percentages
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums
            im = ax.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_xticks(range(len(ACT_TYPE_NAMES)))
            ax.set_yticks(range(len(ACT_TYPE_NAMES)))
            ax.set_xticklabels(ACT_TYPE_NAMES, fontsize=10)
            ax.set_yticklabels(ACT_TYPE_NAMES, fontsize=10)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)
            ax.set_title('Confusion Matrix (row-normalized)', fontsize=14)
            # Annotate cells with count + percentage
            for i in range(len(ACT_TYPE_NAMES)):
                for j in range(len(ACT_TYPE_NAMES)):
                    count = int(last_confusion[i][j])
                    pct = cm_norm[i, j]
                    text = '%d\n(%.1f%%)' % (count, pct * 100) if count > 0 else '0'
                    color = 'white' if pct > 0.5 else 'black'
                    ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            cm_plot_file = os.path.join(run_dir, 'confusion_matrix.png')
            plt.savefig(cm_plot_file, dpi=200)
            plt.close()
            print('Confusion matrix saved:', cm_plot_file)

            # Save confusion matrix as CSV
            cm_csv_file = os.path.join(run_dir, 'confusion_matrix.csv')
            with open(cm_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['true\\pred'] + ACT_TYPE_NAMES)
                for i in range(len(ACT_TYPE_NAMES)):
                    writer.writerow([ACT_TYPE_NAMES[i]] + last_confusion[i])
            print('Confusion matrix CSV saved:', cm_csv_file)

    except ImportError:
        print('matplotlib not installed, skipping plots.')
    except Exception as ex:
        print('Plot failed:', ex)
