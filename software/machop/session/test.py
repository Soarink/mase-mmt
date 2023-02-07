import os
import torch
import pytorch_lightning as pl

from .plt_wrapper import get_model_wrapper


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def test(
        model_name,
        info,
        model, 
        task,
        data_loader, 
        plt_trainer_args, 
        load_path):
    wrapper_cls = get_model_wrapper(model_name, task, info)
    plt_model = wrapper_cls(model)
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, test_dataloaders=data_loader.test_dataloader)
