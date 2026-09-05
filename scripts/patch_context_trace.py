from pathlib import Path


# =====================================================================
# 1. agents/agent.py
# =====================================================================

p = Path("agents/agent.py")
text = p.read_text()

# ---------------------------------------------------------
# Ensure os is imported.
# ---------------------------------------------------------

if "import os\n" not in text:
    lines = text.splitlines(True)

    future_idx = None

    for i, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            future_idx = i
            break

    if future_idx is None:
        lines.insert(
            0,
            "import os\\n"
        )
    else:
        lines.insert(
            future_idx + 1,
            "import os\\n"
        )

    text = "".join(lines)


TRACE_MARKER = '''
        log_prob = log_prob.gather(1, Variable(action))

        self.reward, self.done, self.info = self.episode.step(action[0, 0])
'''

TRACE_REPLACEMENT = '''
        log_prob = log_prob.gather(1, Variable(action))

        # -------------------------------------------------
        # Compact semantic-context trace.
        #
        # Diagnostic only:
        #   - does not modify model_input
        #   - does not modify logits
        #   - does not modify the selected action
        #
        # Enabled by:
        #   AKGVP_SAVE_CONTEXT_TRACE=1
        #
        # Default records steps 0..40 inclusive.
        # -------------------------------------------------
        if os.environ.get(
            "AKGVP_SAVE_CONTEXT_TRACE"
        ) == "1":

            trace_step = len(
                self.episode.actions_record
            )

            trace_max_step = int(
                os.environ.get(
                    "AKGVP_CONTEXT_TRACE_MAX_STEP",
                    "40"
                )
            )

            # New episode.
            if trace_step == 0:
                self.episode.context_trace = []

            if trace_step <= trace_max_step:

                base_attention = getattr(
                    self.model,
                    "last_attention_weight_base",
                    None
                )

                target_embedding = getattr(
                    model_input,
                    "target_class_embedding",
                    None
                )

                if (
                    base_attention is not None
                    and target_embedding is not None
                ):

                    # Full 22-category detector scores
                    # from the current observation.
                    detector_scores = (
                        target_embedding["info"][:, -1]
                    )

                    indicator = (
                        target_embedding["indicator"]
                    )

                    goal_index = int(
                        torch.argmax(
                            indicator.reshape(-1)
                        )
                        .detach()
                        .cpu()
                        .item()
                    )

                    # Semantic-memory KL after the
                    # current forward/update.
                    obj_dist = getattr(
                        self.model,
                        "object_distribution",
                        None
                    )

                    kl_value = getattr(
                        obj_dist,
                        "kl",
                        0.0
                    )

                    if torch.is_tensor(kl_value):
                        semantic_kl = float(
                            kl_value
                            .detach()
                            .reshape(-1)[0]
                            .cpu()
                            .item()
                        )
                    else:
                        semantic_kl = float(
                            kl_value
                        )

                    self.episode.context_trace.append(
                        {
                            "step": int(
                                trace_step
                            ),

                            "state": str(
                                self.episode.states[-1]
                            ),

                            "goal_index": int(
                                goal_index
                            ),

                            "detector_scores": (
                                detector_scores
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "base_attention": (
                                base_attention
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "semantic_kl": (
                                semantic_kl
                            ),

                            "action_probs": (
                                prob
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "action": int(
                                action
                                .detach()
                                .reshape(-1)[0]
                                .cpu()
                                .item()
                            ),
                        }
                    )

        self.reward, self.done, self.info = self.episode.step(action[0, 0])
'''

if "AKGVP_SAVE_CONTEXT_TRACE" not in text:

    if TRACE_MARKER not in text:
        raise RuntimeError(
            "Could not find trace insertion "
            "point in agents/agent.py"
        )

    text = text.replace(
        TRACE_MARKER,
        TRACE_REPLACEMENT,
        1
    )

p.write_text(text)


# =====================================================================
# 2. runners/a3c_val.py
# =====================================================================

p = Path("runners/a3c_val.py")
text = p.read_text()


# ---------------------------------------------------------
# Optional per-room smoke-test limiter.
#
# Unset = exact current evaluation behavior.
# ---------------------------------------------------------

COUNT_MARKER = '''
    targets = AI2THOR_TARGET_CLASSES[args.num_category]
'''

COUNT_REPLACEMENT = '''
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
'''

if "AKGVP_EVAL_MAX_COUNT" not in text:

    if COUNT_MARKER not in text:
        raise RuntimeError(
            "Could not find max_count "
            "insertion point."
        )

    text = text.replace(
        COUNT_MARKER,
        COUNT_REPLACEMENT,
        1
    )


# ---------------------------------------------------------
# Add authoritative full episode histories.
# ---------------------------------------------------------

LOGGER_MARKER = '''
            episode_record["dts"] = float(
                player.episode.done_dis2goal
            )
'''

LOGGER_REPLACEMENT = '''
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
'''

if 'episode_record["full_actions"]' not in text:

    if LOGGER_MARKER not in text:
        raise RuntimeError(
            "Could not find episode logger "
            "insertion point."
        )

    text = text.replace(
        LOGGER_MARKER,
        LOGGER_REPLACEMENT,
        1
    )

p.write_text(text)


print("PATCH COMPLETE")
print()
print("Modified:")
print("  agents/agent.py")
print("  runners/a3c_val.py")
