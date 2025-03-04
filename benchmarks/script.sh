cfg_path=../configs/transformer/llama/llama_86m.json
# cfg_path=../configs/transformer/llama/llama_310m.json
name=llama

# cfg_path=../configs/transformer/mla/mla_86m.json
# name=mla

vocab_size=64000
warmup_steps=16
steps=64

batch_size_list=(16 8 4 2)
seq_len_list=(1024 2048 4096 8192)

# python benchmark_training.py \
#     --cfg_path $cfg_path \
#     --name $name \
#     --vocab_size $vocab_size \
#     --batch_size_list ${batch_size_list[@]} \
#     --seq_len_list ${seq_len_list[@]} \
#     --warmup_steps $warmup_steps \
#     --steps $steps


name=llama_half_rope
cfg_path=../configs/transformer/llama_half_rope/llama_half_rope_90m.json
# cfg_path=../configs/transformer/llama_half_rope/llama_half_rope_310m.json

# name=hgrn2
# cfg_path=../configs/linear_transformer/hgrn2/hgrn2_90m.json
# cfg_path=../configs/linear_transformer/hgrn2/hgrn2_310m.json


vocab_size=16384
batch_size_list=(16)
input_length_list=(1)
# max_length_list=(256)
max_length_list=(576)

python benchmark_inference.py \
    --cfg_path $cfg_path \
    --name $name \
    --vocab_size $vocab_size \
    --batch_size_list ${batch_size_list[@]} \
    --input_length_list ${input_length_list[@]} \
    --max_length_list ${max_length_list[@]} \
    --compile
