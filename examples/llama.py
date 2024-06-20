import torch
from transformers import AutoModel, AutoModelForCausalLM

from xmixers.models import LLaMAConfig

config = LLaMAConfig()

config.update({"num_layers": 12})

model1 = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).cuda()
model2 = AutoModel.from_config(config).to(torch.bfloat16).cuda()

print(config)
print(model1)
print(model2)

b = 2
n = 2048
m = config.vocab_size

input = torch.randint(low=0, high=m, size=(b, n)).cuda()
output1 = model1(input).logits
output2 = model2(input).last_hidden_state
print(output1.shape)
print(output2.shape)
