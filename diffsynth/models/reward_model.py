import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def _checkpoint_dummy_train(*args, **kwargs):
    return None


class _RewardCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "train":
            return _checkpoint_dummy_train
        return super().find_class(module, name)


class _RewardCheckpointPickleModule:
    __name__ = "pickle"
    Unpickler = _RewardCheckpointUnpickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

    @staticmethod
    def load(file, **kwargs):
        return _RewardCheckpointUnpickler(file, **kwargs).load()


def _torch_load_reward_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except AttributeError as exc:
        if "Can't get attribute 'train'" not in str(exc):
            raise
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_RewardCheckpointPickleModule,
        )


class ResnetRewModel(nn.Module):
    def __init__(self, checkpoint_path=None, threshold=None) -> None:
        super().__init__()
        self.threshold = 0.5
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        if threshold is not None:
            self.threshold = float(threshold)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = _torch_load_reward_checkpoint(checkpoint_path)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict)

        if isinstance(checkpoint, dict):
            threshold = checkpoint.get("threshold", None)
            if threshold is None and isinstance(checkpoint.get("metrics"), dict):
                threshold = checkpoint["metrics"].get("best_threshold", None)
            if threshold is not None:
                self.threshold = float(threshold)

    @torch.no_grad()
    def predict_score(self, obs):
        obs = obs.clamp(-1.0, 1.0)
        return self.net(obs.to(dtype=torch.float32))

    @torch.no_grad()
    def predict_rew(self, obs, threshold=None):
        x = self.predict_score(obs)
        threshold = self.threshold if threshold is None else float(threshold)
        return (x >= threshold).to(dtype=x.dtype)

    def forward(self, obs=None):
        return self.predict_rew(obs)
