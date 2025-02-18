date=$(date '+%Y-%m-%d-%H:%M:%S')

folder=models
file=test

model_type=llama
# model_type=mpa
# model_type=tpa
# model_type=hgrn2
model_type=lightnet

dtype=bf16
# dtype=fp32

mkdir -p $folder/log

python $folder/${file}.py --model_type $model_type --dtype $dtype 2>&1 | tee -a $folder/log/${date}-${file}.log
