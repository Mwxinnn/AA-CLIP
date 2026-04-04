import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
from evaluation.evaluator import evaluate_dataset
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_text_adapter(
    adapted_model: nn.Module,
    clip_surgery: nn.Module,
    text_norm_weight: float,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    # scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    text_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            # forward text
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2
            # forward image
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, [6, 12, 18, 24])
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
            # calculate similarity and get prediction
            for f in patch_features:
                # bs,patch_num,768
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss = calculate_seg_loss(patch_preds, mask)
                orthogonal_loss = (
                    (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                    .sum(1)
                    .mean()
                ) ** 2
                loss += orthogonal_loss * text_norm_weight
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )
    return adapted_model


def train_image_adapter(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            B, C, H, W = image.shape
            # forward text
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # forward image
            patch_features, det_feature = model(image)
            # calculate similarity and get prediction
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)
            for f in patch_features:
                # text-image alignment
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)  # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
    return model


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")
    # training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="few_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument(
        "--shot", type=int, default=32, help="number of shots (0 means full shot)"
    )
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--text_epoch", type=int, default=5, help="epochs for stage1")
    parser.add_argument("--image_epoch", type=int, default=20, help="epochs for stage2")
    parser.add_argument(
        "--text_lr", type=float, default=0.00001, help="learning rate for stage1"
    )
    parser.add_argument(
        "--image_lr", type=float, default=0.0005, help="learning rate for stage2"
    )
    parser.add_argument(
        "--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+"
    )
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    # hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    # evaluation
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        default=["MVTec", "Colon_Kvasir", "MPDD"],
        help="datasets used for evaluation",
    )
    parser.add_argument("--batch_size_test", type=int, default=32)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))

    wandb.init(
        project=f"AA-clip-_{args.dataset}_{args.shot}_{args.text_adapt_until}_{args.image_adapt_until}",
        name=f"seed_{args.seed}",
        config=vars(args),
    )

    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # setup image feature extractor after surgery
    clip_surgery = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_surgery.eval()
    clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)
    # set up model for training
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()
    # set optimizer
    text_optimizer = torch.optim.Adam(
        model.text_adapter.parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    image_optimizer = torch.optim.Adam(
        model.image_adapter.parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    # text_scheduler = MultiStepLR(text_optimizer, milestones=[400], gamma=0.1)
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)
    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    logger.info("loading dataset ...")
    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
    )
    text_dataloader = torch.utils.data.DataLoader(
        text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
    )
    logger.info("loading image adaptation dataset ...")
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )
    # ========================================================
    # training
    model = train_text_adapter(
        adapted_model=model,
        clip_surgery=clip_surgery,
        text_norm_weight=args.text_norm_weight,
        train_loader=text_dataloader,
        optimizer=text_optimizer,
        # scheduler=text_scheduler,
        device=device,
        start_epoch=0,
        dataset_name=args.dataset,
        save_path=args.save_path,
        text_epoch=args.text_epoch,
        img_size=args.img_size,
        logger=logger,
    )
    del text_dataloader, text_dataset, clip_surgery, text_optimizer
    torch.cuda.empty_cache()
    with torch.no_grad():
        if args.text_epoch == 0:
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )
        else:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
    model = train_image_adapter(
        model=model,
        text_embeddings=text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=0,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
    )

    # ========================================================
    # evaluation on multiple datasets
    logger.info("Starting evaluation on test datasets...")

    model.eval()

    for test_dataset in args.test_datasets:

        logger.info(f"Evaluating on {test_dataset}")

        df, avg = evaluate_dataset(
            model=model,
            dataset_name=test_dataset,
            img_size=args.img_size,
            batch_size=args.batch_size_test,
            shot=args.shot,
            device=device,
            save_path=args.save_path,
            visualize_flag=args.visualize,
            logger=logger,
        )

        logger.info("\n%s", df.to_string(index=False))

        # log to wandb
        wandb.log(
            {
                f"{test_dataset}/pixel_auc": avg["pixel AUC"],
                f"{test_dataset}/pixel_ap": avg["pixel AP"],
                f"{test_dataset}/image_auc": avg["image AUC"],
                f"{test_dataset}/image_ap": avg["image AP"],
            }
        )


if __name__ == "__main__":
    main()
