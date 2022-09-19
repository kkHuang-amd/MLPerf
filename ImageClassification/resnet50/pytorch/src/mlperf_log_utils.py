import collections
import os
import subprocess
import torch
from mlperf_logging.mllog import constants
from mlperf_logger  import log_event, configure_logger


def mlperf_submission_log(benchmark, submission_platform, num_nodes=1):

    num_nodes = num_nodes

    configure_logger(benchmark)

    log_event(
        key=constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    log_event(
        key=constants.SUBMISSION_ORG,
        value='AMD')

    log_event(
        key=constants.SUBMISSION_DIVISION,
        value='closed')

    log_event(
        key=constants.SUBMISSION_STATUS,
        value='onprem')

    log_event(
        key=constants.SUBMISSION_PLATFORM,
        value=f'{num_nodes}x{submission_platform}')
