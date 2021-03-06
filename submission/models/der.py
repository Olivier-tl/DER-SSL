# https://github.com/aimagelab/mammoth/blob/master/models/der.py

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import typing
import copy

import matplotlib.pyplot as plt
import wandb
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
from sequoia.settings.base.setting import Setting, SettingType
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    Rewards,
    Results,
)
from simple_parsing import ArgumentParser
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import ResNet, resnet18
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from submission.utils.buffer import Buffer
from submission.utils.rotation_transform import Rotation

OUTPUT = 'output'


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
        model_name: bool,
        use_drl: bool,
        drl_lambda: float,
        drl_alpha: float,
        use_data_aug: bool,
        drl_batch_size: int,
    ):
        super().__init__()
        image_space: Image = observation_space.x
        # image_shape = image_space.shape

        # This model is intended for classification / discrete action spaces.
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        self.n_classes = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.encoder, self.representations_size = self.create_encoder(image_space)
        self.output = self.create_output_head(self.n_classes)
        self.ssl_output = self.create_output_head(len(ssl_rotation_angles))
        self.loss = nn.CrossEntropyLoss()
        self.nb_tasks = nb_tasks
        
        # Data Augmentation params
        self.transforms = transforms.Compose([transforms.RandomAffine([-10,10], scale=[0.9,1.1], translate=[0.1,0.1], shear=[-2,2] ),])

        # DER params
        self.alpha = alpha
        self.beta = beta
        self.buffer = Buffer(buffer_size, self.device)
        self.use_data_aug = use_data_aug
        # self.augment = Transform()

        # SSL params
        self.use_ssl = use_ssl
        self.ssl_alpha = ssl_alpha
        self.ssl_m = len(ssl_rotation_angles)
        self.ssl_rotation_angles = ssl_rotation_angles

        # OWM params
        self.use_owm = use_owm

        # DRL params
        self.use_drl = use_drl
        self.drl_lambda = drl_lambda
        self.drl_batch_size = drl_batch_size
        self.drl_alpha = drl_alpha

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
            if self.model_name == 'efficientnet':
                efficent_net = EfficientNet.from_name('efficientnet-b7', in_channels=image_space.c)
                features = efficent_net._fc.in_features
                # Disable/Remove the last layer.
                efficent_net._fc = nn.Sequential()
                encoder = efficent_net
            elif self.model_name == 'resnet':
                # Synbols dataset: use a resnet18 by default.
                resnet: ResNet = resnet18(pretrained=False)
                features = resnet.fc.in_features
                # Disable/Remove the last layer.
                resnet.fc = nn.Sequential()
                encoder = resnet
            else:
                raise ValueError(f'Unknown model name "{self.model_name}"')
        else:
            raise NotImplementedError(f"TODO: Add an encoder for the given image space {image_space}")
        return encoder.to(self.device), features

    def forward(self, observations: Observations) -> Tensor:
        # NOTE: here we don't make use of the task labels.
        observations = observations.to(self.device)
        x = observations.x
        task_labels = observations.task_labels
        features = self.encoder(x)
        logits = self.output(features)
        return logits

    def shared_step(self, batch: Tuple[Observations, Optional[Rewards]],
                    environment: Environment) -> Tuple[Tensor, Dict]:
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

        batch_size = observations.x.shape[0]
        if self.use_data_aug:
            observations = Observations(x=self.transforms(observations.x), task_labels=observations.task_labels)

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

        if self.use_ssl:
            task_id = observations.task_labels[0].item()
            alpha_t = self.alpha * (self.nb_tasks - task_id) / (self.nb_tasks - 1)

            ssl_term = 0
            inputs_ssl = observations.x
            for angle_label in range(self.ssl_m):
                # Rotate data
                angle = self.ssl_rotation_angles[angle_label]
                inputs_transformed = Rotation(angle).forward(inputs_ssl).to(self.device)
                features = self.encoder(inputs_transformed)
                angle_logits = self.ssl_output(features)
                angle_labels = torch.LongTensor([angle_label] * batch_size).to(self.device)
                ssl_term += self.loss(angle_logits, angle_labels)

            loss += alpha_t / self.ssl_m * ssl_term

        # vvvvvv DER vvvvvvv
        if not self.buffer.is_empty():
            # TODO: Add proper transforms argument to get_data().
            #       Actually the DER paper do not use augmentation for
            #       the MNIST dataset so maybe not needed. Issue #8
            buf_inputs, _, buf_logits = self.buffer.get_data(batch_size)
            buf_outputs = self(Observations(x=buf_inputs))
            loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)

            # DER++
            if self.beta != 0:
                buf_inputs, labels, _ = self.buffer.get_data(batch_size)
                buf_outputs = self(Observations(x=buf_inputs))
                loss += self.beta * self.loss(buf_outputs, labels)

            # vvvvvv DRL vvvvvvv
            if self.use_drl:
                _, buf_labels, buf_logits = self.buffer.get_data(self.drl_batch_size)
                while len(torch.unique(buf_labels)) == len(buf_labels):
                    _, buf_labels, buf_logits = self.buffer.get_data(self.drl_batch_size)
                drl_loss_bt = 0
                drl_loss_wi = 0
                n_bt = 0
                n_wi = 0
                for i in range(len(buf_logits)):
                    for j in range(len(buf_logits)):
                        if i == j:
                            continue
                        if buf_labels[i]==buf_labels[j]:
                            n_wi += 1
                            drl_loss_wi += torch.dot(buf_logits[i], buf_logits[j])
                        else:
                            n_bt += 1
                            drl_loss_bt += torch.dot(buf_logits[i], buf_logits[j])
                loss += self.drl_lambda * ((1 / n_bt) * drl_loss_bt + self.drl_alpha * (1/n_wi) * drl_loss_wi)
            # ^^^^^^ DRL ^^^^^^^

        # NOTE: make sure arg examples are not augmented
        self.buffer.add_data(examples=observations.x, logits=logits.data, labels=image_labels)
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
        max_epochs_per_task: int = 15
        # Number of epochs with increasing validation loss after which we stop training.
        early_stop_patience: int = 2

        # Alpha: Weight of logits replay penalty for DER
        alpha: float = 1

        # Beta: Weight of label replay penalty for DER++
        beta: float = 1

        # Buffer size
        buffer_size: int = 5000

        # Use SSL
        use_ssl: bool = True

        # SSL alpha hyperparameter
        ssl_alpha: float = 5

        # List of possible rotation angles for SSL
        ssl_rotation_angles = [0, 90, 180, 270]

        # Use OWM
        use_owm: bool = False

        # Model to use (either efficientnet or resnet)
        model_name: str = 'resnet'

        # Use DRL
        use_drl: bool = False
        drl_lambda: float = 2e-3
        drl_alpha: float = 1
        drl_batch_size: int = 10

        # Use data augmentation
        use_data_aug: bool = True

    def __init__(self, hparams: HParams = None):
        self.hparams: ExampleMethod.HParams = hparams or self.HParams()

        # We will create those when `configure` will be called, before training.
        self.model: DER
        self.optimizer: torch.optim.Optimizer
        # self.trainer_options = TrainerConfig()
        # self.config = Config()

    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        # self.trainer = self.create_trainer(setting)
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
            model_name=self.hparams.model_name,
            use_data_aug=self.hparams.use_data_aug,
            use_drl=self.hparams.use_drl,
            drl_lambda=self.hparams.drl_lambda,
            drl_alpha=self.hparams.drl_alpha,
            drl_batch_size=self.hparams.drl_batch_size,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if not os.path.exists(OUTPUT):
            os.mkdir(OUTPUT)

        wandb.init(project=setting.wandb.project,
                   entity=setting.wandb.entity,
                   mode='offline',
                   dir=OUTPUT,
                   config=dataclasses.asdict(self.hparams))
        self.setting = setting

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        """Training loop.

        NOTE: In the Settings where task boundaries are known (in this case all
        the supervised CL settings), this will be called once per task.
        """
        # # Reset trainer
        # self.trainer = self.create_trainer(self.setting)
        # configure() will have been called by the setting before we get here.
        best_val_loss = inf
        best_model = None
        best_epoch = 0
        for epoch in range(self.hparams.max_epochs_per_task):
            self.model.train()
            print(f"Starting epoch {epoch}")

            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                postfix = {}
                train_pbar.set_description(f"Training Epoch {epoch}")
                epoch_train_loss = 0.0
                train_accuracy = []

                for i, batch in enumerate(train_pbar):
                    loss, metrics_dict = self.model.shared_step(batch, environment=train_env)
                    epoch_train_loss += loss
                    train_accuracy.append(float(metrics_dict['accuracy'][:-1]))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    postfix.update(metrics_dict)
                    train_pbar.set_postfix(postfix)
            train_accuracy = torch.mean(torch.tensor(train_accuracy))
            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                postfix = {}
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.0
                valid_accuracy = []

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(batch, environment=valid_env)
                    epoch_val_loss += batch_val_loss
                    valid_accuracy.append(float(metrics_dict['accuracy'][:-1]))
                    postfix.update(metrics_dict, val_loss=epoch_val_loss)
                    val_pbar.set_postfix(postfix)
            torch.set_grad_enabled(True)
            valid_accuracy = torch.mean(torch.tensor(valid_accuracy))
            wandb.log({
                'epoch': epoch,
                'task_id': self.setting.get_attribute('_current_task_id'),
                'train_accuracy': train_accuracy,
                'valid_accuracy': valid_accuracy,
                'train_loss': epoch_train_loss,
                'valid_loss': epoch_val_loss,
            })
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = i
                best_model_params = self.model.state_dict()
            if i - best_epoch > self.hparams.early_stop_patience:
                print(f"Early stopping at epoch {i}.")
                break
        self.model.load_state_dict(best_model_params)

    def receive_results(self, setting: Setting, results: Results):
        """ Receives the results of an experiment, where `self` was applied to Setting
        `setting`, which produced results `results`.
        """
        wandb.log(results.to_log_dict())
        print(results.make_plots())
        wandb.log(results.make_plots())

    # def create_trainer(self, setting: SettingType) -> Trainer:
    #     """Creates a Trainer object from pytorch-lightning for the given setting.

    #     NOTE: At the moment, uses the KNN and VAE callbacks.
    #     To use different callbacks, overwrite this method.

    #     Args:

    #     Returns:
    #         Trainer: the Trainer object.
    #     """
    #     # We use this here to create loggers!
    #     # callbacks = self.create_callbacks(setting)
    #     loggers = []
    #     # setting.wandb = WandbConfig()
    #     # setting.wandb.project: 'DER-SSL'
    #     # setting.wandb.entity: 'continual-learning'
    #     # setting.wandb.offline: True
    #     if setting.wandb:
    #         wandb_logger = setting.wandb.make_logger('results')
    #         loggers.append(wandb_logger)
    #     trainer = self.trainer_options.make_trainer(
    #         config=self.config, loggers=loggers,
    #     )
    #     return trainer

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    # def create_callbacks(self, setting: SettingType) -> List[Callback]:
    #     """Create the PytorchLightning Callbacks for this Setting.

    #     These callbacks will get added to the Trainer in `create_trainer`.

    #     Parameters
    #     ----------
    #     setting : SettingType
    #         The `Setting` on which this Method is going to be applied.

    #     Returns
    #     -------
    #     List[Callback]
    #         A List of `Callaback` objects to use during training.
    #     """
    #     # TODO: Move this to something like a `configure_callbacks` method in the model,
    #     # once PL adds it.
    #     # from sequoia.common.callbacks.vae_callback import SaveVaeSamplesCallback
    #     return [
    #         # EarlyStopping(monitor="val Loss")
    #         # self.hparams.knn_callback,
    #         # SaveVaeSamplesCallback(),
    #     ]

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

    # HACK: To get the path working
    import sys
    sys.path.insert(0, '../../')

    # Create the Method:

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
