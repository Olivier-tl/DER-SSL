""" DO NOT MODIFY THIS FILE. THIS IS PART OF THE AUTOMATIC EVALUATION!
ONLY MODIFY THE SCRIPT(S) IN ./submission/!
"""
import argparse

import wandb
import fire
from sequoia.methods import Method
from sequoia.settings import (
    ClassIncrementalSetting,
    IncrementalRLSetting,
    Results,
    Setting,
)

from sequoia.client.setting_proxy import SettingProxy
from submission.submission import get_method, get_method_rl, get_method_sl
from submission.models.der import DerMethod


def run_track(method: Method, setting: Setting, yamlfile: str) -> Results:
    setting = SettingProxy(setting, yamlfile)
    results = setting.apply(method)


def run_sl_track(method) -> ClassIncrementalSetting.Results:
    return run_track(method, ClassIncrementalSetting, "sl_track.yaml")


def main(ssl_alpha: float = None, beta: float = None, use_ssl: int = None, model_name: str = None, use_drl: int = None, alpha: float = None, drl_alpha: float = None, drl_lambda: float = None):
    hparams = DerMethod.HParams()
    if ssl_alpha != None:
        hparams.ssl_alpha = ssl_alpha
    if beta != None:
        hparams.beta = beta
    if alpha != None:
        hparams.alpha = alpha
    if use_ssl != None:
        hparams.use_ssl = use_ssl==1
    if model_name != None:
        hparams.model_name = model_name
    if use_drl != None:
        hparams.use_drl = use_drl==1
    if drl_alpha != None:
        hparams.drl_alpha = drl_alpha
    if drl_lambda != None:
        hparams.drl_lambda = drl_lambda
    run_sl_track(get_method_sl(hparams=hparams))


if __name__ == "__main__":
    fire.Fire(main)
