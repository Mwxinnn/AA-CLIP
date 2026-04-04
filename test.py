import os
import argparse
import logging
import torch
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
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


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
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

    # ---- load checkpoints ----
    text_ckpt = os.path.join(args.save_path, "text_adapter.pth")
    image_ckpt = os.path.join(args.save_path, "image_adapter.pth")
    if not os.path.isfile(text_ckpt):
        raise FileNotFoundError(f"Missing {text_ckpt}")
    if not os.path.isfile(image_ckpt):
        raise FileNotFoundError(f"Missing {image_ckpt}")
    text_ckpt = torch.load(text_ckpt, map_location=device)
    model.text_adapter.load_state_dict(text_ckpt["text_adapter"])
    image_ckpt = torch.load(image_ckpt, map_location=device)
    model.image_adapter.load_state_dict(image_ckpt["image_adapter"])
    logger.info("Loaded text_adapter.pth and image_adapter.pth")

    # ========================================================
    # Evaluation
    df, avg = evaluate_dataset(
        model=model,
        dataset_name=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shot=args.shot,
        device=device,
        save_path=args.save_path,
        visualize_flag=args.visualize,
        logger=logger,
    )

    logger.info("\n%s", df.to_string(index=False))
    logging.shutdown()


if __name__ == "__main__":
    main()
