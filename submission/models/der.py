# https://github.com/aimagelab/mammoth/blob/master/models/der.py

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import typing

import gym
import torch
import tqdm
from gym import spaces
from numpy import inf
from sequoia.common.hparams import HyperParameters, log_uniform
from sequoia.common.spaces import Image
from sequoia.methods import Method
from sequoia.settings import ClassIncrementalSetting
from sequoia.settings.passive import PassiveEnvironment
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    Rewards,
)
from simple_parsing import ArgumentParser
from torch import Tensor, nn
from torchvision.models import ResNet, resnet18
from torchvision import transforms
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from submission.utils.buffer import Buffer
from submission.utils.rotation_transform import Rotation



class DER(nn.Module):
    """ Implementation of Dark Experience Replay

    This model uses a resnet18 or efficientNet as the encoder, and a single output layer.
    """

    def __init__(
        self,
        setting: ClassIncrementalSetting,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        nb_tasks: int,
        alpha: float,
        beta: float,
        buffer_size: float,
        use_ssl: bool, 
        ssl_alpha: float,
        ssl_rotation_angles: int,
        use_owm: bool,
    ):
        super().__init__()
        image_space: Image = observation_space.x
        # image_shape = image_space.shape

        # This model is intended for classification / discrete action spaces.
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        self.n_classes = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setting = setting
        self.encoder, self.representations_size = self.create_encoder(image_space)
        self.output = self.create_output_head(self.n_classes)
        self.ssl_output = self.create_output_head(len(ssl_rotation_angles))
        self.loss = nn.CrossEntropyLoss()
        self.nb_tasks = nb_tasks

        # DER params
        self.alpha = alpha
        self.beta = beta
        self.buffer = Buffer(buffer_size, self.device)

        # SSL params
        self.use_ssl = use_ssl
        self.ssl_alpha = ssl_alpha
        self.ssl_m = len(ssl_rotation_angles)
        self.ssl_rotation_angles = ssl_rotation_angles
        
        # OWM params
        self.use_owm = use_owm

    def create_output_head(self, n_outputs) -> nn.Module:
        return nn.Linear(self.representations_size, n_outputs).to(self.device)

    def create_encoder(self, image_space: Image) -> Tuple[nn.Module, int]:
        """Create an encoder for the given image space.

        Returns the encoder, as well as the size of the representations it will produce.

        Parameters
        ----------
        image_space : Image
            A subclass of `gym.spaces.Box` for images. Represents the space the images
            will come from during training and testing. Its attributes of interest
            include `c`, `w`, `h`, `shape` and `dype`.

        Returns
        -------
        Tuple[nn.Module, int]
            The encoder to be used, (a nn.Module), as well as the size of the
            representations it will produce.

        Raises
        ------
        NotImplementedError
            If no encoder is available for the given image dimensions.
        """
        
        # TODO : Add the option for EfficientNet

        if image_space.width == image_space.height == 32:
            # Synbols dataset: use a resnet18 by default.
            resnet: ResNet = resnet18(pretrained=False)
            features = resnet.fc.in_features
            # Disable/Remove the last layer.
            resnet.fc = nn.Sequential()
            encoder = resnet
        else:
            raise NotImplementedError(
                f"TODO: Add an encoder for the given image space {image_space}"
            )
        return encoder.to(self.device), features

    def forward(self, observations: Observations) -> Tensor:
        # NOTE: here we don't make use of the task labels.
        observations = observations.to(self.device)
        x = observations.x
        task_labels = observations.task_labels
        features = self.encoder(x)
        logits = self.output(features)
        return logits

    def shared_step(
        self, batch: Tuple[Observations, Optional[Rewards]], environment: Environment
    ) -> Tuple[Tensor, Dict]:
        """Shared step used for both training and validation.

        Parameters
        ----------
        batch : Tuple[Observations, Optional[Rewards]]
            Batch containing Observations, and optional Rewards. When the Rewards are
            None, it means that we'll need to provide the Environment with actions
            before we can get the Rewards (e.g. image labels) back.

            This happens for example when being applied in a Setting which cares about
            sample efficiency or training performance, for example.

        environment : Environment
            The environment we're currently interacting with. Used to provide the
            rewards when they aren't already part of the batch (as mentioned above).

        Returns
        -------
        Tuple[Tensor, Dict]
            The Loss tensor, and a dict of metrics to be logged.
        """
        # Since we're training on a Passive environment, we will get both observations
        # and rewards, unless we're being evaluated based on our training performance,
        # in which case we will need to send actions to the environments before we can
        # get the corresponding rewards (image labels).
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]
        
        # Get the predictions:
        logits = self(observations)
        y_pred = logits.argmax(-1)

        if rewards is None:
            # If the rewards in the batch is None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            rewards = environment.send(Actions(y_pred))

        assert rewards is not None
        image_labels = rewards.y.to(self.device)

        loss = self.loss(logits, image_labels)

        # vvvvvv DER vvvvvvv

        if (observations.task_labels != observations.task_labels[0]).all().item():
            print('NOT ALL EXAMPLES IN THE BATCH ARE FROM THE SAME TASK, THIS IS NOT CURRENTLY SUPPORTED.')

        if not self.buffer.is_empty():
            batch_size = observations.x.shape[0]
            # TODO: Add proper transforms argument to get_data().
            #       Actually the DER paper do not use augmentation for 
            #       the MNIST dataset so maybe not needed. Issue #8
            buf_inputs, buf_logits = self.buffer.get_data(batch_size) 
            buf_outputs = self(Observations(x=buf_inputs))
            loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)

            ssl_term = 0 
            if self.use_ssl:
                # FIXME: Is it possible that examples in the batch are from different tasks?
                task_id = observations.task_labels[0].item()
                alpha_t = self.alpha * (self.nb_tasks - task_id)/(self.nb_tasks - 1)
                
                ssl_term = 0
                for angle_label in range(self.ssl_m):
                    # Rotate data
                    angle = self.ssl_rotation_angles[angle_label]
                    buf_inputs, _ = self.buffer.get_data(batch_size, transform=Rotation(angle))
                    features = self.encoder(buf_inputs)
                    angle_logits = self.ssl_output(features)
                    angle_labels = torch.LongTensor([angle_label]*batch_size).to(self.device)
                    ssl_term += self.loss(angle_logits, angle_labels)
                
                loss += alpha_t / self.ssl_m * ssl_term

        # NOTE: make sure arg examples are not augmented
        self.buffer.add_data(examples=observations.x, logits=logits.data)
        # ^^^^^^ DER ^^^^^^^
        
        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": f"{accuracy.cpu().item():3.2%}"}
        return loss, metrics_dict


class DerMethod(Method, target_setting=ClassIncrementalSetting):
    """ Method using Dark Experience Replay (DER)

    This method uses the ExampleModel, which is quite simple.
    """

    @dataclass
    class HParams(HyperParameters):
        """ Hyper-parameters of the demo model. """

        # Learning rate of the optimizer.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=0.001)
        # L2 regularization coefficient.
        weight_decay: float = log_uniform(1e-9, 1e-3, default=1e-6)

        # Maximum number of training epochs per task.
        max_epochs_per_task: int = 10
        # Number of epochs with increasing validation loss after which we stop training.
        early_stop_patience: int = 2

        # Alpha: Weight of logits replay penalty for DER
        alpha: float = 0.5

        # Beta: Weight of label replay penalty for DER++
        beta: float = 0.5

        # Buffer size
        buffer_size: int = 500

        # Use SSL
        use_ssl: bool = True

        # SSL alpha hyperparameter 
        ssl_alpha: float = 5

        # List of possible rotation angles for SSL
        ssl_rotation_angles = [0, 90, 180, 270]

        # Use OWM
        use_owm: bool = False

    def __init__(self, hparams: HParams = None):
        self.hparams: ExampleMethod.HParams = hparams or self.HParams()

        # We will create those when `configure` will be called, before training.
        self.model: DER
        self.optimizer: torch.optim.Optimizer

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = DER(
            setting=setting,
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            nb_tasks=setting.nb_tasks,
            alpha=self.hparams.alpha,
            beta=self.hparams.beta,
            buffer_size=self.hparams.buffer_size,
            use_ssl=self.hparams.use_ssl,
            ssl_alpha=self.hparams.ssl_alpha,
            ssl_rotation_angles=self.hparams.ssl_rotation_angles,
            use_owm=self.hparams.use_owm,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """Training loop.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.hparams.max_epochs_per_task):
            self.model.train()
            print(f"Starting epoch {epoch}")
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.shared_step(
                        batch, environment=train_env
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)

            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(
                        batch, environment=valid_env
                    )
                    epoch_val_loss += batch_val_loss
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = i
            if i - best_epoch > self.hparams.early_stop_patience:
                print(f"Early stopping at epoch {i}.")
                # NOTE: You should probably reload the model weights as they were at the
                # best epoch.

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: ExampleMethod.HParams = args.hparams
        return cls(hparams=hparams)


if __name__ == "__main__":
    from sequoia.common import Config
    from sequoia.settings import ClassIncrementalSetting

    # Create the Method:

    # - Manually:
    # method = ExampleMethod()

    # - From the command-line:
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    DerMethod.add_argparse_args(parser)
    args = parser.parse_args()
    method = DerMethod.from_argparse_args(args)

    # - "HARD": Class-Incremental Synbols, more challenging.
    # NOTE: This Setting is very similar to the one used for the SL track of the
    # competition.
    setting = ClassIncrementalSetting(
        dataset="synbols",
        nb_tasks=12,
        known_task_boundaries_at_test_time=False,
        monitor_training_performance=True,
        batch_size=32,
        num_workers=4,
    )
    # NOTE: can also use pass a `Config` object to `setting.apply`. This object has some
    # configuration options like device, data_dir, etc.
    results = setting.apply(method, config=Config(data_dir="data"))
    print(results.summary())

