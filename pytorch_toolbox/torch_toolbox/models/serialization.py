import torch
import os
import torch.nn as nn
import importlib
from typing import Optional, Any, Union, BinaryIO, IO

DEFAULT_PROTOCOL = torch.serialization.DEFAULT_PROTOCOL


def load(
        f: str,
        map_location: Optional[str] = None,
        **kwargs) -> nn.Module:
    """ Load a torch model, either directly or using class specific constructors

    :param f: Path to file
    :param map_location: Target device
    :return: Model

    File f may contain either:

    - a nn.Module directly. In that case, load is equivalent to torch.load
    - a dictionnary with the following mandatory fields:

        - 'class': Indicates the class name of the serialized object
        - 'module': Indicates in which module the class can be found
        - 'loader' (optional): Indicates the name of the loading function. \
                If not provided, then 'load' is used by default.

        In that case, load returns module.class.loader(f,map_location,**kwargs)
    """
    # Load file to CPU first
    model = torch.load(f, 'cpu', **kwargs)

    if isinstance(model, nn.Module):
        return model.to(map_location)

    # Custom deserialization
    if isinstance(model, dict):
        model_dict = model
        if 'module' in model_dict.keys() and 'class' in model_dict.keys():
            module = importlib.import_module(model_dict['module'])
            class_module = module.__getattribute__(model_dict['class'])
            if 'loader' in model_dict.keys():
                loader = getattr(class_module, model_dict['loader'])
            else:
                loader = getattr(class_module, 'load')
            return loader(f, map_location, **kwargs)

    raise ValueError(f'Could not deserialize model file {f}.')


def save(
        obj: Any,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        **kwargs) -> None:
    """ Save a torch model, either directly or using class specific function

    :param obj: Target object
    :param f: Path to file

    - If `obj` contains a `save` method, use it
    - Otherwise, use torch.save
    """
    if hasattr(obj, 'save'):
        obj.save(f, **kwargs)
    else:
        torch.save(obj, f, **kwargs)
