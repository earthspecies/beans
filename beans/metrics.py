import math
import torch

class Accuracy:
    def __init__(self):
        self.num_total = 0
        self.num_correct = 0
    
    def update(self, logits, y):
        self.num_total += logits.shape[0]
        self.num_correct += torch.sum(logits.argmax(axis=1) == y).cpu().item()

    def get_metric(self):
        return {'acc': 0. if self.num_total == 0 else self.num_correct / self.num_total}

    def get_primary_metric(self):
        return self.get_metric()['acc']


class BinaryF1Score:
    def __init__(self):
        self.num_positives = 0
        self.num_trues = 0
        self.num_tps = 0

    def update(self, logits, y):
        positives = logits.argmax(axis=1) == 1
        trues = y == 1
        tps = trues & positives
        self.num_positives += torch.sum(positives).cpu().item()
        self.num_trues += torch.sum(trues).cpu().item()
        self.num_tps += torch.sum(tps).cpu().item()

    def get_metric(self):
        prec = 0. if self.num_positives == 0 else self.num_tps / self.num_positives
        rec = 0. if self.num_trues == 0 else self.num_tps / self.num_trues
        if prec + rec > 0.:
            f1 = 2. * prec * rec / (prec + rec)
        else:
            f1 = 0.

        return {'prec': prec, 'rec': rec, 'f1': f1}

    def get_primary_metric(self):
        return self.get_metric()['f1']


class MulticlassBinaryF1Score:
    def __init__(self, num_classes):
        self.metrics = [BinaryF1Score() for _ in range(num_classes)]
        self.num_classes = num_classes

    def update(self, logits, y):
        probs = torch.sigmoid(logits)
        for i in range(self.num_classes):
            binary_logits = torch.stack((1-probs[:, i], probs[:, i]), dim=1)
            self.metrics[i].update(binary_logits, y[:, i])

    def get_metric(self):
        macro_prec = 0.
        macro_rec = 0.
        macro_f1 = 0.
        for i in range(self.num_classes):
            metrics = self.metrics[i].get_metric()
            macro_prec += metrics['prec']
            macro_rec += metrics['rec']
            macro_f1 += metrics['f1']
        return {
            'macro_prec': macro_prec / self.num_classes,
            'macro_rec': macro_rec / self.num_classes,
            'macro_f1': macro_f1 / self.num_classes
        }

    def get_primary_metric(self):
        return self.get_metric()['macro_f1']


class AveragePrecision:
    """
    Taken from https://github.com/amdegroot/tnt
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.tensor(torch.FloatStorage(), dtype=torch.float32, requires_grad=False)
        self.targets = torch.tensor(torch.LongStorage(), dtype=torch.int64, requires_grad=False)
        self.weights = torch.tensor(torch.FloatStorage(), dtype=torch.float32, requires_grad=False)

    def update(self, output, target, weight=None):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size
                                               + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output.detach())
        self.targets.narrow(0, offset, target.size(0)).copy_(target.detach())

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def get_metric(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
        return ap


class MeanAveragePrecision:
    def __init__(self):
        self.ap = AveragePrecision()

    def reset(self):
        self.ap.reset()

    def update(self, output, target, weight=None):
        self.ap.update(output, target, weight)

    def get_metric(self):
        return {'map': self.ap.get_metric().mean().item()}

    def get_primary_metric(self):
        return self.get_metric()['map']
