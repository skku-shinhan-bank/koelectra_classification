# koelectra_classification
koelectra classification

## Install

```
pip install git+https://github.com/skku-shinhan-bank/koelectra_classification.git
```

## Train

```python
from koelectra_classification import KoElectraClassificationTrainer

train_data_list = [
  'hello', 'hi', 'im', 'shinhan', 'app review'
]
train_label_list = [0, 1, 2, 3, 4]
test_data_list = ['hi', 'hello']
test_label_list = [0, 1]

trainer = KoElectraClassificationTrainer()
trainer.train(
    train_data_list=train_data_list,
    train_label_list=train_label_list,
    test_data_list=test_data_list,
    test_label_list=test_label_list,
    num_of_epochs = 5,
    batch_size = 32,
    num_of_classes = 5,
    max_sequence_length = 128,
    learning_rate = 5e-5,
    max_gradient_normalization = 1,
    warmup_ratio = 0.1,
    model_output_path='output.pth'
)
```

## Predict

```python
from koelectra_classification import KoElectraClassificationPredictor

predictor = KoElectraClassificationPredictor(
    num_of_classes=7,
    model_path='output.pth',
)
predict_class_list = predictor.predict(
    data_list=['hihi', 'heelo', '로그인이 안됨'],
    max_sequence_length=128,
    batch_size=32,
)
```
