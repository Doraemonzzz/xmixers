from transformers import AutoModel

from xmixers.models import LLaMAConfig

config = LLaMAConfig()

print(config)

config.update({"num_layers": 12})

model = AutoModel.from_config(config)

print(model)
