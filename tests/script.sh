date=$(date '+%Y-%m-%d-%H:%M:%S')

folder=models
file=test

model_type=llama

mkdir -p $folder/log

python $folder/${file}.py --model_type $model_type 2>&1 | tee -a $folder/log/${date}-${file}.log
