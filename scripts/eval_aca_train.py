from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval


def main():
    args = command_parser.parse_arguments()

    # Evaluate a frozen navigation checkpoint on the
    # selector-supervision episodes generated exclusively
    # from training scenes and seen target categories.
    args.phase = "eval"
    args.test_or_val = "aca_train"
    args.episode_type = "TestValEpisode"

    print("=" * 80)
    print("ACA SELECTOR TRAIN EVALUATION")
    print("=" * 80)

    print("phase       :", args.phase)
    print("split       :", args.test_or_val)
    print("episode_type:", args.episode_type)
    print("checkpoint  :", args.load_model)

    create_shared_model = model_class(
        args.model
    )

    init_agent = agent_class(
        args.agent_type
    )

    results = main_eval(
        args,
        create_shared_model,
        init_agent,
    )

    print()
    print("=" * 80)
    print("ACA TRAIN EVALUATION FINISHED")
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
