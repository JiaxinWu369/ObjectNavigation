from __future__ import division

import time
import os
import json
import torch
import setproctitle
import copy
import numpy as np
# import gl

from datasets.constants import AI2THOR_TARGET_CLASSES
from datasets.data import name_to_num

from models.model_io import ModelOptions

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
    save_actions,
    write_json,
)


def a3c_val(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
    scenes,
):

    targets = AI2THOR_TARGET_CLASSES[args.num_category]

    # Optional diagnostic smoke-test limiter.
    # Unset = exact original evaluation count.
    max_count_env = os.environ.get(
        "AKGVP_EVAL_MAX_COUNT"
    )

    if max_count_env is not None:
        requested_max_count = int(
            max_count_env
        )

        if requested_max_count <= 0:
            raise ValueError(
                "AKGVP_EVAL_MAX_COUNT must be > 0"
            )

        max_count = min(
            max_count,
            requested_max_count
        )

    if scene_type == "living_room":
        args.max_episode_length = 200
    else:
        args.max_episode_length = 100

    setproctitle.setproctitle("Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        orgin_state = shared_model.state_dict()
        saved_state = torch.load(
            model_to_open, map_location=lambda storage, loc: storage
        )
        orgin_state.update(saved_state)
        shared_model.load_state_dict(orgin_state)

    player = initialize_agent(model_create_fn, args, rank, scenes, targets, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()

    while count < max_count:

        total_reward = 0
        player.eps_len = 0
        new_episode(args, player)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()
        # -------------------------------------------------
        # Case-only qualitative trace.
        #
        # Logging only. eval_index is attached to the Episode
        # so the agent can identify one deterministic case.
        # -------------------------------------------------
        if (
            os.environ.get(
                "AKGVP_CASE_TRACE",
                "0",
            ).strip()
            == "1"
        ):
            player.episode.case_trace_eval_index = int(count)
            player.episode.case_semantic_trace = []
            player.episode.case_execution_trace = []

        # -------------------------------------------------
        # Intervention-frequency diagnostics.
        #
        # Reset once per evaluation episode.
        # -------------------------------------------------
        if (
            os.environ.get(
                "AKGVP_FREQ_DIAG",
                "0",
            ).strip()
            == "1"
        ):
            player.episode.diag_exec_steps = 0
            player.episode.diag_arb_steps = 0
            player.episode.diag_arb_disagree = 0
            player.episode.diag_disagree_use_base = 0
            player.episode.diag_disagree_use_lcr = 0
            player.episode.diag_lcr_selected = 0
            player.episode.diag_br_triggers = 0

        actions = []
        while not player.done:
            player.sync_with_shared(shared_model)
            total_reward = run_episode(player, args, total_reward, model_options, False, shared_model)
            actions = copy.deepcopy([np.squeeze(k.cpu().numpy()).tolist() for k in player.actions])
            if not player.done:
                reset_player(player)

        spl, best_path_length = compute_spl(player, player_start_state)
        # # sxz add
        record_actions = save_actions(actions, player.episode, spl, scene_type, player_start_state)

        # -------------------------------------------------
        # Episode-level paired-evaluation logger.
        #
        # Enabled only when:
        #     AKGVP_SAVE_EPISODES=1
        #
        # eval_index is the deterministic index within
        # the room-specific zs_val split.
        # -------------------------------------------------
        if os.environ.get("AKGVP_SAVE_EPISODES") == "1":
            episode_record = dict(record_actions)

            episode_record["eval_index"] = int(count)
            episode_record["ep_length"] = int(player.eps_len)
            episode_record["dts"] = float(
                player.episode.done_dis2goal
            )

            # Authoritative complete episode histories.
            # Keep legacy "actions" unchanged for
            # backward compatibility.
            episode_record["full_actions"] = [
                int(item)
                for item
                in player.episode.actions_record
            ]

            episode_record["full_states"] = list(
                player.episode.states
            )

            # -------------------------------------------------
            # Compact intervention-frequency diagnostics.
            # -------------------------------------------------
            if (
                os.environ.get(
                    "AKGVP_FREQ_DIAG",
                    "0",
                ).strip()
                == "1"
            ):
                for _diag_key in [
                    "diag_exec_steps",
                    "diag_arb_steps",
                    "diag_arb_disagree",
                    "diag_disagree_use_base",
                    "diag_disagree_use_lcr",
                    "diag_lcr_selected",
                    "diag_br_triggers",
                ]:
                    episode_record[_diag_key] = int(
                        getattr(
                            player.episode,
                            _diag_key,
                            0,
                        )
                    )

            # -------------------------------------------------
            # Optional case-only qualitative trace.
            # -------------------------------------------------
            if (
                os.environ.get(
                    "AKGVP_CASE_TRACE",
                    "0",
                ).strip()
                == "1"
            ):
                episode_record[
                    "case_semantic_trace"
                ] = getattr(
                    player.episode,
                    "case_semantic_trace",
                    [],
                )

                episode_record[
                    "case_execution_trace"
                ] = getattr(
                    player.episode,
                    "case_execution_trace",
                    [],
                )

            # Optional compact semantic trace.
            if os.environ.get(
                "AKGVP_SAVE_CONTEXT_TRACE"
            ) == "1":

                episode_record[
                    "context_trace"
                ] = getattr(
                    player.episode,
                    "context_trace",
                    []
                )

            out_file = os.path.join(
                args.results_path,
                "episodes_{}.jsonl".format(scene_type)
            )

            with open(
                out_file,
                "a",
                encoding="utf-8"
            ) as wf:
                wf.write(
                    json.dumps(
                        episode_record,
                        ensure_ascii=False
                    ) + "\n"
                )

        # gl.app_value('records', record_actions)
        # 计算前进比率
        count_0=0
        if record_actions['success']:
            for action in record_actions['actions']:
                if action==0:
                    count_0+=1
            if count_0==0:
                action_value=0
            else:
                action_value=count_0/float((len(record_actions['actions'])-1))
        else:
            action_value=0

        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success, action_value, player.episode.done_dis2goal)

        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            # DTS=player.episode.done_dis2goal,
            **bucketed_spl,
        )

        count += 1
        reset_player(player)

    player.exit()
    res_queue.put({"END": True})
