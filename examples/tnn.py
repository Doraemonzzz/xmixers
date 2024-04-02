from transformers import AutoModel

from xmixers.models import TnnConfig

config = TnnConfig()

print(config)

config.update({"num_layers": 12})

model = AutoModel.from_config(config)

print(model)
