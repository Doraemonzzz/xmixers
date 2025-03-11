date=$(date '+%Y-%m-%d-%H:%M:%S')

folder=models
file=test

model_type=llama
# model_type=mpa
# model_type=tpa
# model_type=hgrn2
# model_type=lightnet
# model_type=lightnet_scalar_decay
# model_type=lightnet_no_tpe
# model_type=lightnet_no_tpe_scalar_decay
# model_type=mla
# model_type=tnl
# model_type=tnl_state
# model_type=hgrn2_scalar_decay
model_type=linear_transformer
model_type=linear_transformer_no_tpe
model_type=cosformer2
model_type=cosformer2_no_tpe
# model_type=naive_deltanet
# model_type=scalar_decay_deltanet
# model_type=scalar_decay_lower_bound_deltanet
# model_type=vector_decay_deltanet
# model_type=vector_decay_lower_bound_deltanet
# model_type=dense_rnn
# model_type=dense_rnn_lower_bound
model_type=decay_linear_transformer_hgrn2
model_type=decay_linear_transformer_hgrn2_scalar_decay
model_type=decay_linear_transformer_mamba
model_type=decay_linear_transformer_mamba_scalar_decay
model_type=decay_linear_transformer_gla
model_type=decay_linear_transformer_gla_scalar_decay
model_type=decay_linear_transformer_lightnet
model_type=decay_linear_transformer_lightnet_scalar_decay
model_type=decay_linear_transformer_lssp
model_type=decay_linear_transformer_lssp_scalar_decay
model_type=decay_linear_transformer_tnl
model_type=decay_linear_transformer_tnl_scalar_decay

dtype=bf16
# dtype=fp32

mkdir -p $folder/log

python $folder/${file}.py --model_type $model_type --dtype $dtype 2>&1 | tee -a $folder/log/${date}-${file}.log
