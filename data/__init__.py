import numpy as np
import pandas as pd
import h5py
from collections import defaultdict
from sklearn.datasets import make_friedman1
from sksurv.datasets.base import _get_data_path
from sksurv.io import loadarff


def make_survival_data(
        dataset: str
) -> tuple[pd.DataFrame, list[str]]:
    if dataset == "SUPPORT":
        return make_support()
    elif dataset == "METABRIC":
        return make_metabric()
    elif dataset == "SEER_brain":
        return make_seer_brain()
    elif dataset == "SEER_stomach":
        return make_seer_stomach()
    elif dataset == "SEER_liver":
        return make_seer_liver()
    elif dataset == "NACD":
        return make_nacd()
    elif dataset == "DLBCL":
        return make_dlbcl()
    elif dataset == "gbsg":
        return make_gbsg()
    elif dataset == "PBC":
        return make_pbc()
    elif dataset == "VALCT":
        return make_valct()
    elif dataset == "GBM":
        return make_gbm()
    else:
        raise ValueError("Dataset name not recognized.")


def make_support() -> tuple[pd.DataFrame, list[str]]:
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://biostat.app.vumc.org/wiki/Main/SupportDesc for more information.

    Returns
    -------
    pd.DataFrame
        DataFrame with processed covariates for one patient in each row.
    list[str]
        List of columns to standardize.

    References
    ----------
    [1] W. A. Knaus et al., The SUPPORT Prognostic Model: Objective Estimates of Survival
    for Seriously Ill Hospitalized Adults, Ann Intern Med, vol. 122, no. 3, p. 191, Feb. 1995.
    """
    url = "https://biostat.app.vumc.org/wiki/pub/Main/DataSets/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = ["hospdead", "slos", "charges", "totcst", "totmcst", "avtisst", "sfdm2",
                    "adlp", "adls", "dzgroup",  # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
                    # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
                    "sps", "aps", "surv2m", "surv6m", "prg2m", "prg6m", "dnr", "dnrday", "hday"]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    data = (pd.read_csv(url)
            .drop(cols_to_drop, axis=1)
            .rename(columns={"d.time": "time", "death": "event"}))
    data["event"] = data["event"].astype(int)

    data["ca"] = (data["ca"] == "metastatic").astype(int)

    # use recommended default values from official dataset description ()
    # or mean (for continuous variables)/mode (for categorical variables) if not given
    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        "edu": data["edu"].mean(),
        "ph": data["ph"].mean(),
        "glucose": data["glucose"].mean(),
        "scoma": data["scoma"].mean(),
        "meanbp": data["meanbp"].mean(),
        "hrt": data["hrt"].mean(),
        "resp": data["resp"].mean(),
        "temp": data["temp"].mean(),
        "sod": data["sod"].mean(),
        "income": data["income"].mode()[0],
        "race": data["race"].mode()[0],
    }
    data = data.fillna(fill_vals)

    data.sex.replace({'male': 1, 'female': 0}, inplace=True)
    data.income.replace({'under $11k': 0, '$11-$25k': 1, '$25-$50k': 2, '>$50k': 3}, inplace=True)
    skip_cols = ['event', 'sex', 'time', 'dzclass', 'race', 'diabetes', 'dementia', 'ca']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    data = data.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    data.reset_index(drop=True, inplace=True)
    return data, cols_standardize


def make_nacd() -> tuple[pd.DataFrame, list[str]]:
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = pd.read_csv("data/NACD_Full.csv").drop(cols_to_drop, axis=1).rename(columns={"delta": "event"})

    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)
    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    return data, cols_standardize


def make_metabric() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/Metabric.csv").rename(columns={"delta": "event"})
    cols_standardize = ['age_at_diagnosis', 'size', 'lymph_nodes_positive', 'stage', 'lymph_nodes_removed', 'NPI']
    return data, cols_standardize


def make_dlbcl() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/DLBCL.csv").rename(columns={"delta": "event"})
    assert not data.isnull().values.any(), "Dataset contains NaNs"
    skip_cols = ['event', 'time']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_gbsg() -> tuple[pd.DataFrame, list[str]]:
    """
    Rotterdam & German Breast Cancer Study Group (GBSG)

    A combination of the Rotterdam tumor bank and the German Breast Cancer Study Group.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    The original data file 'gbsg_cancer_train_test.h5' is downloaded from
    https://github.com/jaredleekatzman/DeepSurv/blob/master/experiments/data/gbsg/gbsg_cancer_train_test.h5
    """
    data = defaultdict(dict)
    with h5py.File('data/gbsg_cancer_train_test.h5') as f:
        for ds in f:
            for array in f[ds]:
                data[ds][array] = f[ds][array][:]
    train = _make_df(data['train'])
    test = _make_df(data['test'])
    df = pd.concat([train, test]).reset_index(drop=True).rename(columns={"duration": "time"})
    cols_standardize = ['x3', 'x4', 'x5', 'x6']

    del data, train, test
    return df, cols_standardize


def _make_df(data):
    x = data['x']
    t = data['t']
    d = data['e']

    colnames = ['x'+str(i) for i in range(x.shape[1])]
    df = (pd.DataFrame(x, columns=colnames)
          .assign(duration=t)
          .assign(event=d))
    return df


def make_pbc() -> tuple[pd.DataFrame, list[str]]:
    """
    Primary biliary cirrhosis (PBC) of the liver dataset. See [1] and [2] for details.
    [1] https://rdrr.io/cran/survival/man/pbc.html
    [2] T Therneau and P Grambsch (2000), Modeling Survival Data: Extending the Cox Model, Springer-Verlag,
    New York. ISBN: 0-387-98784-3.
    """
    cols_to_drop = ['id']
    data = (pd.read_csv("data/pbc.csv").drop(cols_to_drop, axis=1).
            rename(columns={"status": "event"}))
    # 0/1/2 for censored, transplant, dead
    data.event.replace({1: 0, 2: 1}, inplace=True)
    data.sex.replace({'m': 1, 'f': 0}, inplace=True)
    data.trt = data.trt - 1  # 1/2 -> 0/1 for D-penicillamine, placebo

    fill_vals = {
        "trt": data.trt.mean(),
        "ascites": data.ascites.mode()[0],
        "hepato": data.hepato.mode()[0],
        "spiders": data.spiders.mode()[0],
        "chol": data.chol.mean(),
        "copper": data.copper.mean(),
        "alk.phos": data["alk.phos"].mean(),
        "ast": data.ast.mean(),
        "trig": data.trig.mean(),
        "platelet": data.platelet.mean(),
        "protime": data.protime.mean(),
        "stage": data.stage.mode()[0],
    }

    data = data.fillna(fill_vals)
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['trt', 'sex', 'ascites', 'hepato', 'spiders', 'edema', 'stage', 'event', 'time']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))
    return data, cols_standardize


def make_valct() -> tuple[pd.DataFrame, list[str]]:
    """
    Veterans’ Administration Lung Cancer Trial

    [1] Kalbfleisch, J.D., Prentice, R.L.: “The Statistical Analysis of Failure Time Data.”
    John Wiley & Sons, Inc. (2002)
    """
    fn = _get_data_path("veteran.arff")
    data = loadarff(fn).rename(columns={"Status": "event", "Survival_in_days": "time"})
    data.event.replace({'dead': 1, 'censored': 0}, inplace=True)
    data.Prior_therapy.replace({'no': 0, 'yes': 1}, inplace=True)
    data.Treatment.replace({'standard': 0, 'test': 1}, inplace=True)
    data = pd.get_dummies(data, columns=['Celltype'], drop_first=True)

    cols_standardize = ['Age_in_years', 'Karnofsky_score', 'Months_from_Diagnosis']

    return data, cols_standardize


def make_gbm() -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv("data/GBM.clin.merged.picked.csv").rename(columns={"delta": "event"})
    data.drop(columns=["Composite Element REF", "tumor_tissue_site"], inplace=True)  # Columns with only one value
    data = data[data.time.notna()]  # Unknown censor/event time
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    # Preprocess and fill missing values
    data.gender.replace({'male': 1, 'female': 0}, inplace=True)
    data.radiation_therapy.replace({'yes': 1, 'no': 0}, inplace=True)
    data.ethnicity.replace({'not hispanic or latino': 0, 'hispanic or latino': 1}, inplace=True)
    # one-hot encode categorical variables
    onehot_cols = ["histological_type", "race"]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    fill_vals = {
        "radiation_therapy": data["radiation_therapy"].median(),
        "karnofsky_performance_score": data["karnofsky_performance_score"].median(),
        "ethnicity": data["ethnicity"].median()
    }
    data = data.fillna(fill_vals)
    data.columns = data.columns.str.replace(" ", "_")

    cols_standardize = ['years_to_birth', 'date_of_initial_pathologic_diagnosis', 'karnofsky_performance_score']
    return data, cols_standardize


def make_seer_liver() -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the SEER liver cancer dataset.
    """
    data = pd.read_csv("data/SEER/Liver.csv").rename(columns={"Survival months": "time"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['event', 'time', 'Sex']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    return data, cols_standardize


def make_seer_brain() -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the SEER brain cancer dataset.
    """
    data = pd.read_csv("data/SEER/Brain.csv").rename(columns={"Survival months": "time"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['event', 'time', 'Sex', 'Behavior recode for analysis',
                 'SEER historic stage A (1973-2015)', 'RX Summ--Scope Reg LN Sur (2003+)']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    return data, cols_standardize


def make_seer_stomach() -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the SEER stomach cancer dataset.
    """
    data = pd.read_csv("data/SEER/Stomach.csv").rename(columns={"Survival months": "time"})
    data = data.drop(data[data["time"] <= 0].index)  # remove patients with negative or zero survival time
    data.reset_index(drop=True, inplace=True)

    skip_cols = ['event', 'time', 'Sex']
    cols_standardize = list(set(data.columns.to_list()).symmetric_difference(skip_cols))

    return data, cols_standardize

