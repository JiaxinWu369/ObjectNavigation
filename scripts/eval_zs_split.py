import os

from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval


def main():
    args = command_parser.parse_arguments()

    # Bypass main.py's forced test split.
    args.phase = "eval"
    split = os.environ.get(
        "AKGVP_ZS_SPLIT",
        "zs_val",
    ).strip()

    if not split.startswith("zs_"):
        raise ValueError(
            "AKGVP_ZS_SPLIT must start with zs_"
        )

    args.test_or_val = split
    args.episode_type = "TestValEpisode"

    print("=" * 80)
    print("ZERO-SHOT VALIDATION")
    print("=" * 80)
    print("phase       :", args.phase)
    print("split       :", args.test_or_val)
    print("episode_type:", args.episode_type)
    print("checkpoint  :", args.load_model)

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    results = main_eval(
        args,
        create_shared_model,
        init_agent,
    )

    print("\n" + "=" * 80)
    print("ZS VALIDATION FINISHED")
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
            print(f"{key:28s}: {results[key]}")


if __name__ == "__main__":
    main()
