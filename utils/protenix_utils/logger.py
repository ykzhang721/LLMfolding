# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under Creative Commons Attribution-NonCommercial 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging


__all__ = [
    'get_logger'
]

class WarningFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific warnings by their message or other criteria
        if "simtk.openmm" in record.getMessage():
            return False
        return True


def get_logger(name="", loglevel="INFO", log_file_path=None):
    root_logger = logging.getLogger()
    logger = logging.getLogger(name)
    # we only add handlers to the root logger! Let the propogation handle the rest.
    add_handlers(root_logger, loglevel, log_file_path)
    return logger


def add_handlers(logger, loglevel, log_file_path=None):
    fmt = "%(asctime)-15s [%(pathname)s:%(lineno)d] %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)
    loglevel = getattr(logging, loglevel.upper(), logging.INFO)
    logger.setLevel(loglevel)

    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    else:
        handler = logger.handlers[0]
    handler.setFormatter(formatter)
    warning_filter = WarningFilter()
    handler.addFilter(warning_filter)

    # we output to at most two streams: one stdout and one file
    if log_file_path is not None and len(logger.handlers) == 1:
        handler = logging.FileHandler(log_file_path, mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler.addFilter(warning_filter)

    return logger
