from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.adam import Adam
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa, ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher3DLoss, VonMisesFisher2DLoss
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
from graphnet.utilities.logging import get_logger
from typing import Any, Dict, List, Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model.IceCubeModel import  IceCubeModel
import pandas as pd
import numpy as np
import os

logger = get_logger()

def build_model(config: Dict[str, Any], train_dataloader: Any) -> IceCubeModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
    )

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        prediction_columns = [config["target"] + "_x",
                              config["target"] + "_y",
                              config["target"] + "_z",
                              config["target"] + "_kappa"]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = IceCubeModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader) / 2,
                len(train_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-02, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes

    return model


def load_pretrained_model(config: Dict[str, Any],
                          state_dict_path: str = '/kaggle/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth') -> IceCubeModel:
    train_dataloader, _ = make_dataloaders(config=config)
    model = build_model(config=config,
                        train_dataloader=train_dataloader)
    # model._inference_trainer = Trainer(config['fit'])
    model.load_state_dict(state_dict_path)
    model.prediction_columns = [config["target"] + "_x",
                                config["target"] + "_y",
                                config["target"] + "_z",
                                config["target"] + "_kappa"]
    model.additional_attributes = ['zenith', 'azimuth', 'event_id']
    return model


def make_dataloaders(config: Dict[str, Any]) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""
    train_dataloader = make_dataloader(db=config['path'],
                                       selection=pd.read_csv(config['train_selection'])[
                                           config['index_column']].ravel().tolist(),
                                       pulsemaps=config['pulsemap'],
                                       features=features,
                                       truth=truth,
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       shuffle=True,
                                       labels={'direction': Direction()},
                                       index_column=config['index_column'],
                                       truth_table=config['truth_table'],
                                       )

    validate_dataloader = make_dataloader(db=config['path'],
                                          selection=pd.read_csv(config['validate_selection'])[
                                              config['index_column']].ravel().tolist(),
                                          pulsemaps=config['pulsemap'],
                                          features=features,
                                          truth=truth,
                                          batch_size=config['batch_size'],
                                          num_workers=config['num_workers'],
                                          shuffle=False,
                                          labels={'direction': Direction()},
                                          index_column=config['index_column'],
                                          truth_table=config['truth_table'],

                                          )
    return train_dataloader, validate_dataloader



def main(config):

    """Builds and trains GNN according to config."""
    logger.info(f"features: {config['features']}")
    logger.info(f"truth: {config['truth']}")

    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"

    train_dataloader, validate_dataloader = make_dataloaders(config=config)

    model = build_model(config, train_dataloader)


    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ModelCheckpoint(monitor="val_loss"),
        ProgressBar(),
    ]

    t_logger = TensorBoardLogger(
        'tensorboard_logs',config['name'], default_hp_metric=False
    )

    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks = callbacks,
        logger = t_logger,
        **config["fit"],
    )
    # do inference
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    # Make Dataloader
    test_dataloader = make_dataloader(db=config['inference_database_path'],
                                      selection=None,  # Entire database
                                      pulsemaps=config['pulsemap'],
                                      features=features,
                                      truth=truth,
                                      batch_size=config['batch_size'],
                                      num_workers=config['num_workers'],
                                      shuffle=False,
                                      labels={'direction': Direction()},
                                      index_column=config['index_column'],
                                      truth_table=config['truth_table'],
                                      )

    # Get predictions
    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=model.additional_attributes,
    )
    # Save predictions and model to file
    archive = os.path.join(config['base_dir'], "train_model_without_configs")
    run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"
    db_name = config['path'].split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")


if __name__ == '__main__':
    # Constants
    features = FEATURES.KAGGLE
    truth = TRUTH.KAGGLE

    # Configuration
    config = {
        "name": 'IceCube_Default',
        "path": '/home/yicong/working/batch_1.db',
        "inference_database_path": '/home/yicong/working/batch_1.db',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": 200,
        "num_workers": 24,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
            "max_epochs": 50,
            "gpus": [0],
            "distribution_strategy": None,
        },
        'train_selection': 'Dataset/train_selection_max_200_pulses.csv',
        'validate_selection': 'Dataset/validate_selection_max_200_pulses.csv',
        'test_selection': None,
        'base_dir': 'training'
    }
    main(config)