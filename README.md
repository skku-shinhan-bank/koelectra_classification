# koelectra_classification
koelectra classification

## Install

```
pip install git+https://github.com/skku-shinhan-bank/koelectra_classification.git
```

## Train

```python
from koelectra_classification import KoElectraClassificationTrainer
import torch

train_data = [
  'hello', 'hi', 'im', 'shinhan', 'app review'
]
train_label = [0, 1, 2, 3, 4]
test_data = ['hi', 'hello']
test_label = [0, 1]

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

trainer = KoElectraClassificationTrainer()
trainer.train(
    train_data=train_data,
    train_label=train_label,
    test_data=test_data,
    test_label=test_label,
    num_of_epochs = 5,
    batch_size = 32,
    num_of_classes = 5,
    max_sequence_length = 128,
    learning_rate = 5e-5,
    max_gradient_normalization = 1,
    warmup_ratio = 0.1,
    device=device,
    model_output_path='output.pth'
)
```
