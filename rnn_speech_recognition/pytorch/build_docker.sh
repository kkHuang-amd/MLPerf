#!/usr/bin/env bash

IMAGE_TAG=rnnt_train
docker build --network=host -t $IMAGE_TAG --pull -f Dockerfile .
