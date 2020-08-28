#!/bin/sh
#
# Move cluster output to bkp dir

data_dir="data/triton"
bkp_dir="data/bkp"

[ -d $bkp_dir ] || mkdir -p $bkp_dir
mv $data_dir/* $bkp_dir && echo "Data backed up successfully!" || echo "Error backing up data"
