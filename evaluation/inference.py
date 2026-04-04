import numpy as np
import torch
from tqdm import tqdm

from forward_utils import calculate_similarity_map
from dataset.constants import DOMAINS


def get_predictions(
    model,
    class_text_embeddings,
    test_loader,
    device,
    img_size,
    dataset_name,
):

    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []

    for batch in tqdm(test_loader):

        image = batch["image"].to(device)
        mask = batch["mask"].cpu().numpy()
        label = batch["label"].cpu().numpy()
        file_name = batch["file_name"]
        class_name = batch["class_name"]

        assert len(set(class_name)) == 1

        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)

        patch_feats, img_feat = model(image)

        # image-level score
        img_pred = img_feat @ class_text_embeddings
        img_pred = (img_pred[:, 1] + 1) / 2
        preds_image.append(img_pred.cpu().numpy())

        # pixel-level score
        patch_preds = []

        for f in patch_feats:
            patch_pred = calculate_similarity_map(
                f,
                class_text_embeddings,
                img_size,
                test=True,
                domain=DOMAINS[dataset_name],
            )

            patch_preds.append(patch_pred)

        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()

        preds.append(patch_preds)

    return (
        np.concatenate(masks),
        np.concatenate(labels),
        np.concatenate(preds),
        np.concatenate(preds_image),
        file_names,
    )
