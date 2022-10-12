import pytest
import torch
from beans.metrics import Accuracy, BinaryF1Score, MulticlassBinaryF1Score, AveragePrecision, MeanAveragePrecision

def test_accuracy():
    metric = Accuracy()
    assert metric.get_metric() == {'acc': 0.0}
    metric.update(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0]]))
    assert metric.get_metric() == {'acc': 1.0}
    metric.update(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[1]]))
    assert metric.get_metric() == {'acc': 1.0}
    metric.update(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[2]]))
    assert metric.get_metric() == {'acc': 1.0}
    metric.update(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[3]]))     # incorrect prediction
    assert metric.get_metric() == {'acc': 0.75}

def test_binary_f1score():
    metric = BinaryF1Score()
    metric.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[1]]))      # true positive
    assert metric.get_metric() == {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([[0]]))      # true negative
    assert metric.get_metric() == {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    metric.update(torch.tensor([[0.0, 1.0]]), torch.tensor([[0]]))      # false positive
    assert metric.get_metric() == {'prec': 0.5, 'rec': 1.0, 'f1': 2/3}
    metric.update(torch.tensor([[1.0, 0.0]]), torch.tensor([[1]]))      # false negative
    assert metric.get_metric() == {'prec': 0.5, 'rec': 0.5, 'f1': 0.5}

def test_multiclass_binary_f1score():
    metric = MulticlassBinaryF1Score(num_classes=4)
    # tp, tn, fp, fn
    metric.update(torch.tensor([[1.0, -1.0, 1.0, -1.0]]), torch.tensor([[1, 0, 0, 1]]))
    assert metric.metrics[0].get_metric() == {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    assert metric.metrics[1].get_metric() == {'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
    assert metric.metrics[2].get_metric() == {'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
    assert metric.metrics[3].get_metric() == {'prec': 0.0, 'rec': 0.0, 'f1': 0.0}
    assert metric.get_metric() == {'macro_prec': 0.25, 'macro_rec': 0.25, 'macro_f1': 0.25}

    metric.update(torch.tensor([[1.0, 1.0, 1.0, 1.0]]), torch.tensor([[1, 1, 1, 1]]))
    assert metric.metrics[0].get_metric() == {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    assert metric.metrics[1].get_metric() == {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    assert metric.metrics[2].get_metric() == {'prec': 0.5, 'rec': 1.0, 'f1': 2/3}
    assert metric.metrics[3].get_metric() == {'prec': 1.0, 'rec': 0.5, 'f1': 2/3}
    assert metric.get_metric() == pytest.approx({'macro_prec': 7/8, 'macro_rec': 7/8, 'macro_f1': 5/6})

def test_mean_average_precision():
    metric = AveragePrecision()

    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([0.1, 0.2, 0.3, 4])
    weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])
    metric.update(output, target, weight)

    ap = metric.get_metric()
    val = (1*0.1/0.1 + 0*2.0/2.1 + 1.1*1/3.1 + 0*1/4)/2.0
    assert ap[0] == pytest.approx(val)

    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([0, 1, .5, .5])
    metric.reset()
    metric.update(output, target)
    assert metric.get_metric()[0] == pytest.approx((1 + 2/3)/2)

    metric = MeanAveragePrecision()
    target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
    output = torch.Tensor([[0.1, 0.2, 0.3, 4], [0, 1, .5, .5]]).transpose(0, 1)
    metric.update(output, target)
    assert metric.get_metric()['map'] == pytest.approx((1 + 2/3)/2)
