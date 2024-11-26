import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    train_test_split as sk_train_test_split,
)

RANDOM_STATE = 42


def get_hm27k_data():
    tumor_meth_data = pd.read_csv(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/methylation/hm27_hm450_merge_meth_data.tsv",
        sep="\t",
    )

    normal_meth_data = pd.read_csv(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/methylation/normals/normal_samples_hm27.tsv",
        sep="\t",
    )
    tumor_formatted = tumor_meth_data.set_index("ENTITY_STABLE_ID").drop(
        columns=["NAME", "DESCRIPTION", "TRANSCRIPT_ID"]
    )
    normal_formatted = normal_meth_data.set_index(tumor_formatted.index).drop(
        columns=["chrom", "start", "end"]
    )
    normal_formatted = normal_formatted / 100
    return tumor_formatted, normal_formatted


def separate_tumor_normal():
    samples_methylation = pd.read_feather(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/methylation/hm450Kmeth_data.feather",
    )
    samples_methylation_meta = pd.read_csv(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/methylation/hm450Kmeth_metadata.tsv",
        sep="\t",
    )
    cancer_methylation_meta = samples_methylation_meta[
        samples_methylation_meta["cases__samples__sample_type"] == "Primary Tumor"
    ]
    normal_methylation_meta = samples_methylation_meta[
        samples_methylation_meta["cases__samples__sample_type"] == "Solid Tissue Normal"
    ]
    cancer_methylation = samples_methylation[
        cancer_methylation_meta["cases__samples__submitter_id"]
    ].copy()
    cancer_methylation["id"] = samples_methylation["id"]
    cancer_methylation = cancer_methylation[
        ["id"] + [col for col in cancer_methylation.columns if col != "id"]
    ]
    normal_methylation = samples_methylation[
        normal_methylation_meta["cases__samples__submitter_id"]
    ].copy()
    normal_methylation["id"] = samples_methylation["id"]
    normal_methylation = normal_methylation[
        ["id"] + [col for col in normal_methylation.columns if col != "id"]
    ]
    return (
        cancer_methylation,
        cancer_methylation_meta,
        normal_methylation,
        normal_methylation_meta,
    )


def get_merged_normal_data():
    (
        _,
        _,
        normal_methylation,
        normal_methylation_meta,
    ) = separate_tumor_normal()

    supplemental_normal_methylation = pd.read_csv(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/methylation/normals/normal_samples_450K.tsv",
        sep="\t",
    )

    supplemental_normal_methylation = supplemental_normal_methylation[
        ["key"]
        + [col for col in supplemental_normal_methylation.columns if col != "key"]
    ]
    supplemental_normal_info = pd.read_csv(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/methylation/normals/normal_samples_info.tsv",
        sep="\t",
    )

    normal_methylation = normal_methylation.merge(
        supplemental_normal_methylation,
        left_on="id",
        right_on="key",
    )
    normal_methylation.drop(columns=["key"], inplace=True)

    # merged meta data
    column_coversions = {
        "cases__samples__sample_id": "sample_id",
        "cases__submitter_id": "sample",
        "cases__project__project_id": "project_id",
        "cases__samples__sample_type": "sample_type",
    }

    normal_methylation_meta = normal_methylation_meta.rename(columns=column_coversions)
    # drop columns that are not in columns_conversions values
    normal_methylation_meta = normal_methylation_meta[
        [
            col
            for col in normal_methylation_meta.columns
            if col in column_coversions.values()
        ]
    ]
    supplemental_normal_info = supplemental_normal_info[
        [
            col
            for col in supplemental_normal_info.columns
            if col in column_coversions.values()
        ]
    ]

    normal_methylation_meta = pd.concat(
        [normal_methylation_meta, supplemental_normal_info], ignore_index=True, axis=0
    )
    return normal_methylation, normal_methylation_meta


def load_full_data():
    (
        cancer_methylation,
        cancer_methylation_meta,
        _,
        _,
    ) = separate_tumor_normal()

    normal_methylation, normal_methylation_meta = get_merged_normal_data()

    cancer_clinical = pd.read_csv(
        "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20230828_tcga_methylation/side_projects/20232110_TCGA_METHYLATION_CLINICAL_ML/data/clinical/clinical_patient.tsv",
        sep="\t",
    )

    data_dict = {
        "cancer_methylation": cancer_methylation,
        "cancer_methylation_meta": cancer_methylation_meta,
        "normal_methylation": normal_methylation,
        "normal_methylation_meta": normal_methylation_meta,
        "cancer_clinical": cancer_clinical,
    }

    return data_dict


def get_merged_meta_data():
    data = load_full_data()
    # merged meta data
    column_coversions = {
        "cases__samples__sample_id": "sample_id",
        "cases__submitter_id": "sample",
        "cases__project__project_id": "project_id",
        "cases__samples__sample_type": "sample_type",
    }

    data["cancer_methylation_meta"] = data["cancer_methylation_meta"].rename(
        columns=column_coversions
    )
    # drop columns that are not in columns_conversions values
    data["cancer_methylation_meta"] = data["cancer_methylation_meta"][
        [
            col
            for col in data["cancer_methylation_meta"].columns
            if col in column_coversions.values()
        ]
    ]
    return pd.concat(
        [
            data["cancer_methylation_meta"],
            data["normal_methylation_meta"],
        ],
        ignore_index=True,
        axis=0,
    )


def get_tumor_normal_ml_data(hm_27=False):
    if hm_27:
        tumor_methylation, normal_methylation = get_hm27k_data()
        data = {
            "cancer_methylation": tumor_methylation,
            "normal_methylation": normal_methylation,
        }
    else:
        data = load_full_data()

        data["cancer_methylation"].set_index("id", inplace=True)
        data["normal_methylation"].set_index("id", inplace=True)
    X = pd.concat(
        [
            data["cancer_methylation"],
            data["normal_methylation"],
        ],
        axis=1,
    )
    y = pd.concat(
        [
            pd.Series([1] * data["cancer_methylation"].shape[1]),
            pd.Series([0] * data["normal_methylation"].shape[1]),
        ],
        ignore_index=True,
        axis=0,
    )
    return X.T, y


def get_subtype_ml_data(brca_only=True):
    data = load_full_data()
    data["cancer_methylation"].set_index("id", inplace=True)
    data["normal_methylation"].set_index("id", inplace=True)
    X = pd.concat(
        [
            data["cancer_methylation"],
            data["normal_methylation"],
        ],
        axis=1,
    )
    clinical_data = data["cancer_clinical"]
    sample_subtypes = []
    for sample in X.columns:
        submitter_id = sample[:-4]
        if submitter_id not in clinical_data["PATIENT_ID"].values:
            sample_subtype = "Not Available"
        else:
            sample_subtype = clinical_data[clinical_data["PATIENT_ID"] == submitter_id][
                "SUBTYPE"
            ].values[0]
        sample_subtypes.append(sample_subtype)
    y = pd.DataFrame(sample_subtypes, index=X.columns, columns=["subtype"])
    data = X.copy().T
    data["subtype"] = y

    if brca_only:
        data.dropna(subset=["subtype"], inplace=True)
        data = data[data["subtype"].str.startswith("BRCA")]

    X = data.drop(columns=["subtype"])
    y = data["subtype"]

    return X, y


def train_test_split(X, y):
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def stratified_train_test_split(X, y):
    X_values = X.values
    y_values = y.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    splits = sss.split(X_values, y_values)
    train_index, test_index = next(splits)
    X_train, X_test, y_train, y_test = (
        X_values[train_index],
        X_values[test_index],
        y_values[train_index],
        y_values[test_index],
    )
    X_train = pd.DataFrame(X_train, columns=X.columns, index=X.index[train_index])
    X_test = pd.DataFrame(X_test, columns=X.columns, index=X.index[test_index])
    y_train = pd.Series(y_train, index=y.index[train_index])
    y_test = pd.Series(y_test, index=y.index[test_index])
    return X_train, X_test, y_train, y_test
