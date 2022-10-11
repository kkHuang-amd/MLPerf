#!/bin/bash
set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/raid/datasets/coco/coco-2017}"
: "${LOGDIR:=$(pwd)/results}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=object_detection_run
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results")

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME} / ${DGX_PRETTY_NAME:-???} ${DGX_OTA_VERSION:-${DGX_SWBUILD_VERSION:-???}}"
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
source "${_config_file}"
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    #docker container rm -f "${_cont_name}" || true
    docker stop "${_cont_name}" || true 
    docker container rm -f object_detection_debug || true
    docker rename "${_cont_name}" object_detection_debug || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
#nvidia-docker run --privileged --init --detach \
docker run --privileged --init --detach --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /global/home/mtsai/workspace/logs/:/logs \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        docker exec -it "${_cont_name}" python -c "
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.mlperf_logger import log_event
log_event(constants.MASKRCNN)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
#            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.mlperf_logger import log_event
log_event(key=constants.CACHE_CLEAR, value=True, stack_offset=0)"
        fi
        #enter interactive mode
        #docker exec -it "${_config_env[@]}" "${_cont_name}" bash 
        # Run experiment 
        #docker exec -it "${_config_env[@]}" "${_cont_name}" rocprof --stats -o /logs/output_20221004.csv ./run_and_time.sh
        #docker exec -it "${_config_env[@]}" "${_cont_name}" /workspace1/cmd_docker.sh
        docker exec -it "${_config_env[@]}" "${_cont_name}" ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
