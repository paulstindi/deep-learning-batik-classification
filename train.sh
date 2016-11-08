#!/bin/bash

now="$(date +'%Y%m%d%H%M')"
logfile="train.$now.log"

if [[ $1 -eq "--gpu" ]]; then
    time THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py dataset.h5 dataset.index.json > $logfile 2>&1 &
else
    time python train.py dataset.h5 dataset.index.json > $logfile 2>&1 &
fi

tail -100f $logfile
