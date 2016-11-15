#!/bin/bash

now="$(date +'%Y%m%d%H%M')"
logfile="preprocess.$now.log"

time python preprocess.py ~/BatikTesting dataset.h5 dataset.test.h5 dataset.index.json > $logfile 2>&1 &

#tail -100f $logfile
