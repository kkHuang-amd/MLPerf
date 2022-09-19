import argparse
import torch


def add_parser_arguments(parser):
    parser.add_argument(
        "--checkpoint-path", metavar="<path>", help="checkpoint filename"
    )
    parser.add_argument(
        "--weight-path", metavar="<path>", help="name of file in which to store weights"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    add_parser_arguments(parser)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)

    model_state_dict = {
        k[len("module.1.") :] if "module.1." in k else k: v
        for k, v in checkpoint["state_dict"].items()
    }

    print(f"Loaded {checkpoint['arch']} : {checkpoint['best_prec1']}")

    torch.save(model_state_dict, args.weight_path)
