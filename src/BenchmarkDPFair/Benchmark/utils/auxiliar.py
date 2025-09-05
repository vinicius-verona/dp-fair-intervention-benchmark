import pandas as pd
import numpy as np
import os

def getMetrics(metric):

    DI         = metric.disparate_impact() if metric is not None else None
    ACC        = metric.accuracy() if metric is not None else None
    ACC_PRIV   = metric.accuracy(privileged=True) if metric is not None else None
    ACC_UNPRIV = metric.accuracy(privileged=False) if metric is not None else None
    PREC       = metric.precision() if metric is not None else None
    REC        = metric.recall() if metric is not None else None
    MAD        = metric.accuracy(privileged=False) - metric.accuracy(privileged=True) if metric is not None else None
    EOD        = metric.equal_opportunity_difference() if metric is not None else None
    TPR        = metric.true_positive_rate() if metric is not None else None
    FPR        = metric.false_positive_rate() if metric is not None else None
    TNR        = metric.true_negative_rate() if metric is not None else None
    FNR        = metric.false_negative_rate() if metric is not None else None
    SPD        = metric.statistical_parity_difference() if metric is not None else None
    EODD       = metric.equalized_odds_difference() if metric is not None else None

    if metric is not None:
        del metric

    return {
        "DI": DI if (DI is not None and not np.isnan(DI)) else 'inf',
        "ACC": ACC if (ACC is not None and not np.isnan(ACC)) else 'inf',
        "ACC_PRIV": ACC_PRIV if (ACC_PRIV is not None and not np.isnan(ACC_PRIV)) else 'inf',
        "ACC_UNPRIV": ACC_UNPRIV if (ACC_UNPRIV is not None and not np.isnan(ACC_UNPRIV)) else 'inf',
        "PREC": PREC if (PREC is not None and not np.isnan(PREC)) else 'inf',
        "REC": REC if (REC is not None and not np.isnan(REC) ) else 'inf',
        "MAD": MAD if (MAD is not None and not np.isnan(MAD) ) else 'inf',
        "EOD": EOD if (EOD is not None and not np.isnan(EOD) ) else 'inf',
        "TPR": TPR if (TPR is not None and not np.isnan(TPR) ) else 'inf',
        "FPR": FPR if (FPR is not None and not np.isnan(FPR) ) else 'inf',
        "TNR": TNR if (TNR is not None and not np.isnan(TNR) ) else 'inf',
        "FNR": FNR if (FNR is not None and not np.isnan(FNR) ) else 'inf',
        "SPD": SPD if (SPD is not None and not np.isnan(SPD) ) else 'inf',
        "EODD": EODD if (EODD is not None and not np.isnan(EODD) ) else 'inf',
    }

def save_experiment(experiment, seed, eps=None, filename="exp_metrics.csv", path="../data/metrics/", synth=""):
    results = []
    exp_set_original  = "original_classification_metrics"
    exp_set_mitigator = "mitigated_classification_metrics"

    logs = []


    for r in experiment.results:
        if "error" in r:
            logs.append({
                "Seed": seed,
                "Epsilon": eps if eps is not None else "",
                "Fair-Method": r["mitigator"] if "mitigator" in r else "",
                "DP-Method": synth if "dp_method" in r else "",
                "Error": r["error"],
                "Info": r["info"]
            })
            r.pop("error", None)
            r.pop("info", None)

        for exp_set in [exp_set_original, exp_set_mitigator]:
            if exp_set in r:
                if r[exp_set] is None: 
                    continue

                results.append({
                    "Seed": seed,
                    "Epsilon": eps if eps is not None else "",
                    "Fair-Method": r["mitigator"] if "mitigator" in r else "",
                    "DP-Method": synth if "dp_method" in r else "",
                    **(r[exp_set]),
                })

    del experiment.results

    # Check if file exists
    file_exists = os.path.isfile(path+filename)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save results to CSV
    pd.DataFrame(results).to_csv(path + filename, index=False, mode='a', header=not file_exists)

    # Check if file exists
    file_exists = os.path.isfile(path+"log/"+filename.replace(".csv", "-log.csv"))

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path + "log/"), exist_ok=True)

    # Save logs to CSV
    pd.DataFrame(logs).to_csv(path + "log/" + filename.replace(".csv", "-log.csv"), index=False, mode='a', header=not file_exists)