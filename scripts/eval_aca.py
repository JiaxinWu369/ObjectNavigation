import os

from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval


def main():

    args = command_parser.parse_arguments()

    split = os.environ.get(
        "AKGVP_ACA_SPLIT",
        "aca_train",
    ).strip()

    if not split.startswith("aca_"):
        raise ValueError(
            "AKGVP_ACA_SPLIT must start with aca_"
        )

    args.phase = "eval"
    args.test_or_val = split
    args.episode_type = "TestValEpisode"

    print("=" * 80)
    print("ACA EVALUATION")
    print("=" * 80)

    print("phase       :", args.phase)
    print("split       :", args.test_or_val)
    print("episode_type:", args.episode_type)
    print("checkpoint  :", args.load_model)

    results = main_eval(
        args,
        model_class(args.model),
        agent_class(args.agent_type),
    )

    print()
    print("=" * 80)
    print("ACA EVALUATION FINISHED")
    print("=" * 80)

    for key in [
        "success",
        "spl",
        "ep_length",
        "done_count",
        "GreaterThan/1/success",
        "GreaterThan/1/spl",
        "GreaterThan/1/dts",
        "GreaterThan/5/success",
        "GreaterThan/5/spl",
        "GreaterThan/5/dts",
    ]:
        if key in results:
            print(
                f"{key:28s}: "
                f"{results[key]}"
            )


if __name__ == "__main__":
    main()
