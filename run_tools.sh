#!/usr/bin/env bash

tensorboard --logdir /data/summary/flowers/ &

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --notebook-dir='/notebooks' "$@"
