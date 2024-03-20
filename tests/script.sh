date=$(date '+%Y-%m-%d-%H:%M:%S')

folder=ops
file=long_conv_1d

mkdir -p $folder/log

pytest $folder/${file}.py  2>&1 | tee -a $folder/log/${date}-${file}.log
