from pyeer.eer_info import get_eer_stats


def calculate_metrics(gen_scores, fake_scores, epoch):
    metrics = get_eer_stats(gen_scores, fake_scores)
    return metrics.fmr0, metrics.fmr100, metrics.fmr1000, metrics.gmean, metrics.imean, metrics.auc
