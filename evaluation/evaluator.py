import torch
from pandas import DataFrame, Series
from torch.utils.data import DataLoader

from dataset import get_dataset
from dataset.constants import DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    metrics_eval,
    visualize,
)

from .inference import get_predictions


def evaluate_dataset(
    model,
    dataset_name,
    img_size,
    batch_size,
    shot,
    device,
    save_path,
    visualize_flag=False,
    logger=None,
):

    kwargs = {"num_workers": 4, "pin_memory": True} if device.type == "cuda" else {}

    datasets = get_dataset(
        dataset_name,
        img_size,
        None,
        shot,
        "test",
        logger=logger,
    )

    with torch.no_grad():
        text_embeddings = get_adapted_text_embedding(model, dataset_name, device)

    df = DataFrame(
        columns=["class name", "pixel AUC", "pixel AP", "image AUC", "image AP"]
    )

    for class_name, dataset in datasets.items():

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        )

        with torch.no_grad():
            masks, labels, preds, preds_image, file_names = get_predictions(
                model,
                text_embeddings[class_name],
                loader,
                device,
                img_size,
                dataset_name,
            )

        if visualize_flag:
            visualize(
                masks,
                preds,
                file_names,
                save_path,
                dataset_name,
                class_name=class_name,
            )

        df.loc[len(df)] = Series(
            metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[dataset_name],
            )
        )

    metric_cols = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
    df[metric_cols] = df[metric_cols].astype(float)

    avg = df[metric_cols].mean().to_dict()
    avg["class name"] = "Average"
    df.loc[len(df)] = Series(avg)

    return df, avg
