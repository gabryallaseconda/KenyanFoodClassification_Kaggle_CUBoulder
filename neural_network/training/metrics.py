import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import torch
import matplotlib.pyplot as plt


def get_loss():
    return torch.nn.CrossEntropyLoss()

from neural_network.configuration import metricsConfig


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count



    # def update(self, val, count=1):
    #     self.val = val
    #     self.sum += val * count
    #     self.count += count
    #     self.avg = self.sum / self.count





class AccuracyEstimator():
    def __init__(self, topk=(1, 2, 3, 5, )):
        self.topk = topk
        self.metrics = [AverageMeter() for i in range(len(topk) + 1)]

    def reset(self):
        for i in range(len(self.metrics)):
            self.metrics[i].reset()

    def update_value(self, pred, target):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i, k in enumerate(self.topk):
                correct_k = correct[:k].reshape(-1).float().sum()
                self.metrics[i].update(correct_k.item() * (100.0 / batch_size))
                #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                #self.metrics[i].update(correct_k.mul_(100.0 / batch_size).item())

    def get_metric_value(self):
        metrics = {}
        for i, k in enumerate(self.topk):
            metrics["top{}".format(k)] = self.metrics[i].avg
        return metrics




class ConfusionMatrixMeter:
    """
    Accumula la confusion matrix per problemi multiclass.
    update() accetta logits o label predette.
    """
    def __init__(self, num_classes: int, device: torch.device | str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self):
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor, from_logits: bool = True):
        """
        preds: [B, C] logits (from_logits=True) oppure [B] class indices (from_logits=False)
        targets: [B] class indices
        """
        if from_logits:
            preds = preds.argmax(dim=1)
        preds = preds.view(-1).to(self.device)
        targets = targets.view(-1).to(self.device)

        # bincount su indici combinati: t*C + p
        k = self.num_classes * targets + preds
        binc = torch.bincount(k, minlength=self.num_classes**2)
        self.cm += binc.view(self.num_classes, self.num_classes)

    def get_cm(self) -> torch.Tensor:
        return self.cm.clone()

    def get_cm_numpy(self) -> np.ndarray:
        return self.cm.detach().cpu().numpy()

    def plot_figure(self, class_names: list[str] | None = None, normalize: bool = False):
        """
        Ritorna una figura matplotlib della confusion matrix.
        normalize=True -> normalizza per riga (recall per classe).
        """
        cm = self.get_cm_numpy().astype(float)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm = cm / row_sums

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        n = self.num_classes
        ax.set(
            xticks=np.arange(n),
            yticks=np.arange(n),
            xticklabels=class_names if class_names else np.arange(n),
            yticklabels=class_names if class_names else np.arange(n),
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix' + (' (normalized)' if normalize else '')
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # valori nelle celle
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
        fmt = '.2f' if normalize else 'd'
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                text = f"{val:{fmt}}" if (normalize or val != 0) else ""
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if (val > thresh) else "black")

        fig.tight_layout()
        return fig




class PRF1Meter:
    """
    Calcola precision/recall/F1 per classe + macro/micro/weighted a partire dalla Confusion Matrix.
    Usa ConfusionMatrixMeter oppure passa direttamente la CM.
    """
    @staticmethod
    def from_confusion_matrix(cm: torch.Tensor | np.ndarray):
        if isinstance(cm, torch.Tensor):
            cm = cm.detach().cpu().numpy()
        cm = cm.astype(np.int64)

        tp = np.diag(cm)
        pred_pos = cm.sum(axis=0)  # colonne
        true_pos = cm.sum(axis=1)  # righe
        support = true_pos

        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_class = np.divide(tp, pred_pos, where=pred_pos > 0)
            recall_per_class = np.divide(tp, true_pos, where=true_pos > 0)
            f1_per_class = np.divide(2 * precision_per_class * recall_per_class,
                                     precision_per_class + recall_per_class,
                                     where=(precision_per_class + recall_per_class) > 0)

        # macro
        macro_precision = np.nanmean(precision_per_class)
        macro_recall = np.nanmean(recall_per_class)
        macro_f1 = np.nanmean(f1_per_class)

        # micro (su tutti i campioni)
        total_tp = tp.sum()
        total_pred_pos = pred_pos.sum()
        total_true_pos = true_pos.sum()
        micro_precision = total_tp / total_pred_pos if total_pred_pos > 0 else 0.0
        micro_recall = total_tp / total_true_pos if total_true_pos > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if (micro_precision + micro_recall) > 0 else 0.0)

        # weighted (pesato per support)
        weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support, dtype=float)
        weighted_precision = np.nansum(precision_per_class * weights)
        weighted_recall = np.nansum(recall_per_class * weights)
        weighted_f1 = np.nansum(f1_per_class * weights)

        return {
            "per_class": {
                "precision": precision_per_class.tolist(),
                "recall": recall_per_class.tolist(),
                "f1": f1_per_class.tolist(),
                "support": support.tolist(),
            },
            "macro": {"precision": float(macro_precision), "recall": float(macro_recall), "f1": float(macro_f1)},
            "micro": {"precision": float(micro_precision), "recall": float(micro_recall), "f1": float(micro_f1)},
            "weighted": {"precision": float(weighted_precision), "recall": float(weighted_recall), "f1": float(weighted_f1)},
        }


def global_grad_norm(model: torch.nn.Module) -> float:
    """Norma L2 globale dei gradienti (solo parametri con grad)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            total += g.pow(2).sum().item()
    return math.sqrt(total)

def iterate_weight_and_grad_tensors(model: torch.nn.Module):
    """
    Generatore per loggare istogrammi su TensorBoard.
    Yield: (name, weight_tensor, grad_tensor_or_None)
    """
    for name, p in model.named_parameters():
        w = p.data.detach()
        g = p.grad.detach() if (p.grad is not None) else None
        yield name, w, g


class ThroughputLatencyMeter:
    """
    Tiene traccia di:
    - total_images
    - total_time (s)
    - batches (per latency media)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_images = 0
        self.total_time = 0.0
        self.total_batches = 0

    def update(self, batch_size: int, batch_time_seconds: float):
        self.total_images += batch_size
        self.total_time += batch_time_seconds
        self.total_batches += 1

    def get_throughput_img_per_s(self) -> float:
        return self.total_images / self.total_time if self.total_time > 0 else 0.0

    def get_latency_ms_per_batch(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return (self.total_time / self.total_batches) * 1000.0

    def get_epoch_time_seconds(self) -> float:
        return self.total_time


class GPUMemoryMeter:
    """
    Legge il picco di memoria GPU allocata tra reset() e get().
    Ricorda di chiamare torch.cuda.reset_peak_memory_stats() all'inizio dell'epoca.
    """
    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device) if isinstance(device, str) else device

    def reset(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_memory_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        peak = torch.cuda.max_memory_allocated(self.device)  # bytes
        return peak / (1024 ** 3)




########################################################################################3333333


import torch


def metric_epoch_orchestrator(predictions: torch.Tensor,
                           targets: torch.Tensor):
    
    metrics = {}

    if metricsConfig.is_accuracy_enabled:
        metrics.update(_metric_epoch_accuracy(predictions=predictions, targets=targets))

    if metricsConfig.is_confusion_matrix_enabled:
        metrics.update(_metric_epoch_confusion_matrix(predictions=predictions, targets=targets))

    if metricsConfig.is_precision_recall_enabled:
        metrics.update(_metric_epoch_precision_recall(predictions=predictions, targets=targets))

    return metrics

def _metric_epoch_accuracy(predictions: torch.Tensor,
                           targets: torch.Tensor):

    topk=metricsConfig.accuracy_topk

    maxk = max(topk)
    _, pred_topk = predictions.topk(maxk, dim=1, largest=True, sorted=True)  # [N, maxk]
    target_exp = targets.view(-1, 1).expand_as(pred_topk)
    correct = pred_topk.eq(target_exp)  # [N, maxk]

    res = {}
    for k in topk:
        correct_k = correct[:, :k].any(dim=1).float().sum().item()
        acc = correct_k / targets.size(0) * 100.0
        res[f"accuracy_top{k}"] = acc
    return res


def _metric_epoch_confusion_matrix(predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   class_names=None):
    # predizione di classe
    y_pred = predictions.argmax(dim=1)
    num_classes = predictions.size(1)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        assert len(class_names) == num_classes, \
            f"class_names ha len={len(class_names)} ma num_classes={num_classes}"

    # confusion matrix
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1

    cm_plot = cm.clone().to(dtype=torch.float32)
    if metricsConfig.confusion_matrix_normalize:
        row_sums = cm_plot.sum(dim=1, keepdim=True).clamp(min=1)
        cm_plot = cm_plot / row_sums

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        cm_plot.numpy(),
        interpolation="nearest",
        aspect="equal",
        vmin=0.0 if metricsConfig.confusion_matrix_normalize else None,
        vmax=1.0 if metricsConfig.confusion_matrix_normalize else None,
    )
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    # sposta il labelpad qui:
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)

    ax.set_title(
        "Confusion Matrix (normalized)"
        if metricsConfig.confusion_matrix_normalize else
        "Confusion Matrix"
    )

    # migliora leggibilità delle xticks se i nomi sono lunghi
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # annotazioni
    thresh = cm_plot.max().item() / 2.0 if cm_plot.numel() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_plot[i, j].item()
            text = f"{val:.2f}" if metricsConfig.confusion_matrix_normalize else f"{int(val)}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    fig.tight_layout()

    return {
        "confusion_matrix_fig": fig,
        # "confusion_matrix_tensor": cm
    }

def _metric_epoch_precision_recall(predictions: torch.Tensor,
                                   targets: torch.Tensor):
    y_pred = predictions.argmax(dim=1)
    num_classes = predictions.size(1)

    # confusion matrix
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1

    eps = 1e-12

    # per-class
    TP = cm.diag().to(torch.float64)
    FP = cm.sum(dim=0).to(torch.float64) - TP
    FN = cm.sum(dim=1).to(torch.float64) - TP
    support = cm.sum(dim=1).to(torch.float64)  

    prec_per_class = TP / (TP + FP + eps)
    rec_per_class  = TP / (TP + FN + eps)
    f1_per_class   = 2 * prec_per_class * rec_per_class / (prec_per_class + rec_per_class + eps)

    # aggregate
    if metricsConfig.precision_recall_aggregation == "macro":
        precision = prec_per_class.mean()
        recall = rec_per_class.mean()
        f1 = f1_per_class.mean()
    elif metricsConfig.precision_recall_aggregation == "weighted":
        weights = support / (support.sum() + eps)
        precision = (prec_per_class * weights).sum()
        recall = (rec_per_class * weights).sum()
        f1 = (f1_per_class * weights).sum()
    elif metricsConfig.precision_recall_aggregation == "micro":
        TP_sum = TP.sum()
        FP_sum = FP.sum()
        FN_sum = FN.sum()
        precision = TP_sum / (TP_sum + FP_sum + eps)
        recall = TP_sum / (TP_sum + FN_sum + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
    else:
        raise ValueError("average must be one of: 'macro', 'micro', 'weighted'")

    if metricsConfig.precision_recall_as_percent:
        precision = precision.item() * 100.0
        recall = recall.item() * 100.0
        f1 = f1.item() * 100.0
    else:
        precision = precision.item()
        recall = recall.item()
        f1 = f1.item()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "per_class": {
        #     "precision": prec_per_class.tolist(),
        #     "recall": rec_per_class.tolist(),
        #     "f1": f1_per_class.tolist(),
        # }
    }


