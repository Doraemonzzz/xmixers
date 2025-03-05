vocab_size=16384
batch_size_list=(16)
input_length_list=(1)
max_length_list=(256)
max_length_list=(576)

model_type=linear_transformer
name=hgrn2
for config_name in hgrn2_xxl #hgrn2_90m #hgrn2_310m hgrn2_xl hgrn2_xxl

# model_type=transformer
# name=llama_half_rope
# for config_name in llama_half_rope_90m #llama_half_rope_310m llama_half_rope_xl llama_half_rope_xxl
do
    cfg_path=../configs/${model_type}/${name}/${config_name}.json
    date=$(date '+%Y-%m-%d-%H:%M:%S')
    python benchmark_inference.py \
        --cfg_path $cfg_path \
        --name $name \
        --vocab_size $vocab_size \
        --batch_size_list ${batch_size_list[@]} \
        --input_length_list ${input_length_list[@]} \
        --max_length_list ${max_length_list[@]} \
        --compile \
        --cg \
        2>&1 | tee -a logs/${date}-${config_name}.log
    # --cg
done
