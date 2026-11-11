from collections import OrderedDict
from aif360.metrics import ClassificationMetric

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()

    # performance metrics
    metrics["true_negative_rate"] = classified_metric_pred.specificity()
    metrics["true_positive_rate"] = classified_metric_pred.sensitivity()
    metrics["false_positive_rate"] = classified_metric_pred.false_positive_rate()
    metrics["false_negative_rate"] = classified_metric_pred.false_negative_rate()
    metrics["precision"] = classified_metric_pred.precision()
    metrics["false_discovery_rate"] = classified_metric_pred.false_discovery_rate()
    metrics["false_omission_rate"] = classified_metric_pred.false_omission_rate()
    metrics["accuracy"] = classified_metric_pred.accuracy()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())

    # classification metrics
    metrics["error_rate"] = classified_metric_pred.error_rate()
    metrics["average_odds_difference"] = classified_metric_pred.average_odds_difference()
    metrics["average_abs_odds_difference"] = classified_metric_pred.average_abs_odds_difference()
    metrics["selection_rate"] = classified_metric_pred.selection_rate()
    metrics["disparate_impact"] = classified_metric_pred.disparate_impact()
    metrics["statistical_parity_difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["generalized_entropy_index"] = classified_metric_pred.generalized_entropy_index()
    metrics["theil_index"] = classified_metric_pred.theil_index()
    metrics["equal_opportunity_difference"] = classified_metric_pred.equal_opportunity_difference()
    
    # if disp:
    #     for k in metrics:
    #         print("%s = %.4f" % (k, metrics[k]))
    
    return metrics