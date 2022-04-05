"""
Utility functions for all tests

Created: April 5, 2022
"""
__author__ = "Ramsey Karim"

import os
import socket
import pwd

import scoby

import scoby.config
DEBUG_SHOW = True
debug_img_path = scoby.config.debug_img_path


def create_debug_img_metadata(file=None, func_name=None) -> dict:
    """
    Create a PNG-appropriate metadata dictionary
    which will be passed to matplotlib's savefig function
    :param file: the __file__ variable. Easy, just pass it through.
    :param func_name: name of the current function. This is not trivial
        to get automatically, so just pass it in here
    :returns: dict, appropriate to provide PNG metadata
    """
    source = []
    if file is not None:
        source.append(os.path.basename(file).replace('.py', ''))
    if func_name is not None:
        source.append(func_name)
    source = '.'.join(source)
    if not source:
        source = 'unspecified location'
    source = f'({pwd.getpwuid(os.getuid())[0]}@{socket.gethostname()}) {source} (scoby v{scoby.__version__})'
    return {"Title": "func_name", "Source": source}
