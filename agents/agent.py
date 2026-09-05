""" Base class for all Agents. """
from __future__ import division
import os

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from datasets.constants import DONE_ACTION_INT, AI2THOR_TARGET_CLASSES


class ThorAgent:
    """ Base class for all actor-critic agents. """

    def __init__(
        self, model, args, rank, scenes, targets, episode=None, max_episode_length=1e3, gpu_id=-1
    ):
        self.scenes = scenes
        self.targets = targets
        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[22]) if item in self.targets]

        self.gpu_id = gpu_id

        self._model = None
        self.model = model
        self._episode = episode
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = False
        self.info = None
        self.reward = 0
        self.max_length = False
        self.hidden = None
        self.actions = []
        self.probs = []
        self.last_action_probs = None
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.max_episode_length = max_episode_length
        self.success = False
        self.backprop_t = 0
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)

        self.verbose = args.verbose
        self.learned_loss = args.learned_loss
        self.learned_input = None
        self.learned_t = 0
        self.num_steps = args.num_steps
        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space

        self.targets_types = None
        self.model_name = args.model

        self.action_num = 0
        self.meta_learning_actions = {}
        self.meta_predictions = []
        self.meta_duplicate_action = args.meta_duplicate_action
        self.meta_failed_action = args.meta_failed_action
        self.meta_all_steps = args.meta_all_steps

        self.memory_duplicate_learning = args.memory_duplicate_learning
        self.duplicate_states_actions = {}

        # imitation learning related parameters
        self.imitation_learning = args.imitation_learning
        self.il_duplicate_action = args.il_duplicate_action
        self.il_failed_action = args.il_failed_action
        self.il_each_action = args.il_each_action
        self.il_update_actions = {}

        self.record_attention = args.record_attention

        # depth related parameters
        self.depth = args.depth
        self.depth_maximum = args.depth_maximum
        
        #zsx add
        self.model_phase = args.model_phase

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self, model_options):
        """ Eval at state. """
        raise NotImplementedError()

    @property
    def episode(self):
        """ Return the current episode. """
        return self._episode

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def state(self):
        """ Return the state of the agent. """
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    @property
    def model(self):
        """ Returns the model. """
        return self._model

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    @model.setter
    def model(self, model_to_set):
        self._model = model_to_set
        if self.gpu_id >= 0 and self._model is not None:
            with torch.cuda.device(self.gpu_id):
                self._model = self.model.cuda()

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False

    def action(self, model_options, training, test_update):
        """ Train the agent. """
        if training or test_update:
            self.model.train()
        else:
            self.model.eval()

        self.episode.states.append(str(self.episode.environment.controller.state))

        model_input, out = self.eval_at_state(model_options)

        # record the output of the model
        self.hidden = out.hidden

        if out.state_representation is not None:
            self.episode.state_reps.append(out.state_representation.squeeze().cpu())
        if out.state_memory is not None:
            self.episode.state_memory.append(out.state_memory.squeeze().cpu())
        if out.action_memory is not None:
            self.episode.action_memory.append(out.action_memory.squeeze().cpu())
        if out.obs_rep is not None:
            self.episode.obs_reps.append(out.obs_rep.squeeze().cpu())

        if out.meta_action is not None:
            self.meta_predictions.append(F.softmax(out.meta_action, dim=1))
            self.episode.meta_predictions.append(F.softmax(out.meta_action, dim=1))

        if out.match_score is not None and self.record_attention:
            visual_info = {
                'match_score': out.match_score,
            }
            self.episode.match_score.append(visual_info['match_score'])

        # agent operates the asked action
        prob = F.softmax(out.logit, dim=1)
        self.episode.action_outputs.append(prob.tolist())
        if training:
            action = prob.multinomial(1).data
        else:
            action = prob.argmax(dim=1, keepdim=True)

        log_prob = F.log_softmax(out.logit, dim=1)
        self.last_action_probs = prob
        entropy = -(log_prob * prob).sum(1)
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

                            "lcr_stats": (
                                dict(
                                    getattr(
                                        self,
                                        "_last_lcr_stats",
                                        {},
                                    )
                                )
                                if getattr(
                                    self,
                                    "_last_lcr_stats",
                                    None,
                                ) is not None
                                else None
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

        # ============================================================
        # Blocked Recovery
        #
        # Detect consecutive failed MoveAhead actions from unchanged
        # simulator states. Once the streak reaches K, if the policy
        # again selects MoveAhead, execute its highest-probability
        # alternative action while excluding MoveAhead and Done.
        #
        # Disabled by default:
        #   AKGVP_BLOCK_RECOVERY=1
        #   AKGVP_BLOCK_STREAK=5
        # ============================================================

        if (
            not hasattr(self, "_block_stall_streak")
            or self.action_num == 0
        ):
            self._block_stall_streak = 0

        _br_enabled = (
            os.environ.get(
                "AKGVP_BLOCK_RECOVERY",
                "0",
            ).strip() == "1"
        )

        _br_k = int(
            os.environ.get(
                "AKGVP_BLOCK_STREAK",
                "5",
            )
        )

        _br_debug = (
            os.environ.get(
                "AKGVP_BLOCK_DEBUG",
                "0",
            ).strip() == "1"
        )

        # Whether Blocked Recovery also suppresses Done.
        # Default=1 exactly reproduces the original BR behavior.
        _br_mask_done = (
            os.environ.get(
                "AKGVP_BLOCK_MASK_DONE",
                "1",
            ).strip() == "1"
        )

        # Recovery strategy used only after the blocked-MoveAhead
        # condition has been triggered.
        #
        # policy:
        #   remove MoveAhead and preserve the current policy ranking
        #   over the remaining admissible actions.
        #
        # fixed_left / fixed_right:
        #   diagnostic ablations used only on the validation split.
        _br_mode = os.environ.get(
            "AKGVP_BLOCK_RECOVERY_MODE",
            "policy",
        ).strip().lower()

        if _br_mode not in {
            "policy",
            "fixed_left",
            "fixed_right",
        }:
            raise ValueError(
                "Unknown AKGVP_BLOCK_RECOVERY_MODE: "
                + _br_mode
            )

        _br_original_action = int(
            action.detach()
            .reshape(-1)[0]
            .cpu()
            .item()
        )

        _br_executed_action = (
            _br_original_action
        )

        # Case-trace bookkeeping only.
        _br_stall_before = int(
            self._block_stall_streak
        )
        _br_triggered = False

        # Recovery is activated only when:
        #   1) the previous K MoveAhead actions all failed, and
        #   2) the policy again wants MoveAhead.
        if (
            _br_enabled
            and self._block_stall_streak >= _br_k
            and _br_original_action == 0
        ):
            _br_triggered = True

            # =============================================
            # BR activation diagnostic.
            #
            # Logging only. Entering this branch means
            # the safeguard actually constrains the
            # nominal MoveAhead action.
            # =============================================
            if (
                os.environ.get(
                    "AKGVP_FREQ_DIAG",
                    "0",
                ).strip()
                == "1"
            ):
                self.episode.diag_br_triggers = (
                    int(
                        getattr(
                            self.episode,
                            "diag_br_triggers",
                            0,
                        )
                    )
                    + 1
                )

            _br_probs = (
                prob.detach()
                .reshape(-1)
            )

            # Keep direction selection policy-driven.
            # Exclude:
            #   0 = MoveAhead
            #   DONE_ACTION_INT = Done
            _br_excluded = {0}

            if _br_mask_done:
                _br_excluded.add(
                    int(DONE_ACTION_INT)
                )

            _br_candidates = [
                i
                for i in range(
                    int(_br_probs.numel())
                )
                if i not in _br_excluded
            ]

            if _br_mode == "fixed_left":
                # AI2-THOR action mapping:
                # 1 = RotateLeft
                _br_alt = 1

            elif _br_mode == "fixed_right":
                # AI2-THOR action mapping:
                # 2 = RotateRight
                _br_alt = 2

            else:
                # Original policy-ranked recovery:
                # exclude the empirically invalidated MoveAhead action
                # and preserve the current policy ranking over all
                # remaining admissible actions.
                _br_alt = max(
                    _br_candidates,
                    key=lambda i:
                        float(
                            _br_probs[i]
                            .detach()
                            .cpu()
                            .item()
                        ),
                )

            action = action.clone()
            action[0, 0] = _br_alt

            _br_executed_action = int(
                _br_alt
            )

            if _br_debug:
                print(
                    "[BLOCK-RECOVERY]",
                    "scene=",
                    self.episode.environment.scene_name,
                    "target=",
                    self.episode.target_object,
                    "step=",
                    self.action_num,
                    "streak=",
                    self._block_stall_streak,
                    "original=",
                    _br_original_action,
                    "recovery=",
                    _br_executed_action,
                    "prob=",
                    round(
                        float(
                            _br_probs[_br_alt]
                            .detach()
                            .cpu()
                            .item()
                        ),
                        5,
                    ),
                    flush=True,
                )

        # State immediately before the actually executed action.
        _br_state_before = str(
            self.episode
            .environment
            .controller
            .state
        )

        self.reward, self.done, self.info = self.episode.step(action[0, 0])

        # Exact number of actions actually executed in AI2-THOR.
        # Logging only.
        if (
            os.environ.get(
                "AKGVP_FREQ_DIAG",
                "0",
            ).strip()
            == "1"
        ):
            self.episode.diag_exec_steps = (
                int(
                    getattr(
                        self.episode,
                        "diag_exec_steps",
                        0,
                    )
                )
                + 1
            )

        # State immediately after execution.
        _br_state_after = str(
            self.episode
            .environment
            .controller
            .state
        )

        if _br_enabled:
            if (
                _br_executed_action == 0
                and
                _br_state_after == _br_state_before
            ):
                self._block_stall_streak += 1
            else:
                self._block_stall_streak = 0

        # =================================================
        # Case-only execution trace.
        #
        # Recorded after environment execution so the
        # before/after simulator states are authoritative.
        # =================================================
        if (
            os.environ.get(
                "AKGVP_CASE_TRACE",
                "0",
            ).strip()
            == "1"
        ):
            _case_scene = os.environ.get(
                "AKGVP_CASE_SCENE",
                "",
            ).strip()

            _case_target = os.environ.get(
                "AKGVP_CASE_TARGET",
                "",
            ).strip()

            _case_eval_index = int(
                os.environ.get(
                    "AKGVP_CASE_EVAL_INDEX",
                    "-1",
                )
            )

            _case_match = bool(
                str(
                    self.episode
                    .environment
                    .scene_name
                )
                == _case_scene
                and
                str(
                    self.episode
                    .target_object
                )
                == _case_target
                and
                int(
                    getattr(
                        self.episode,
                        "case_trace_eval_index",
                        -999999,
                    )
                )
                == _case_eval_index
            )

            if _case_match:
                self.episode.case_execution_trace.append(
                    {
                        "step": int(
                            self.action_num
                        ),

                        "state_before":
                            _br_state_before,

                        "state_after":
                            _br_state_after,

                        "stall_before":
                            int(
                                _br_stall_before
                            ),

                        "stall_after":
                            int(
                                self._block_stall_streak
                            ),

                        "nominal_action":
                            int(
                                _br_original_action
                            ),

                        "executed_action":
                            int(
                                _br_executed_action
                            ),

                        "br_triggered":
                            bool(
                                _br_triggered
                            ),

                        "policy_probs": (
                            prob.detach()
                            .reshape(-1)
                            .cpu()
                            .tolist()
                        ),
                    }
                )

        self.action_num += 1

        # record the actions those should be used to compute the loss
        meta_update_step = False
        if self.episode.meta_learning:
            if self.meta_duplicate_action:
                if str(self.episode.environment.controller.state) in self.episode.states:
                    meta_update_step = True
            elif self.meta_failed_action:
                if not self.info:
                    meta_update_step = True
            elif self.meta_all_steps:
                meta_update_step = True

        if meta_update_step:
            optimal_action = self.environment.controller.get_optimal_action(self.episode.target_object)
            self.meta_learning_actions[self.action_num - 1] = optimal_action

        # record the actions those should be used to compute the loss during imitation learning
        imitation_learning_update = False
        if self.imitation_learning:
            if self.il_duplicate_action and str(self.episode.environment.controller.state) in self.episode.states:
                imitation_learning_update = True
            elif self.il_failed_action and not self.info:
                imitation_learning_update = True
            elif self.il_each_action:
                imitation_learning_update = True

        if imitation_learning_update:
            optimal_action = self.environment.controller.get_optimal_action(self.episode.target_object)
            self.il_update_actions[self.action_num-1] = optimal_action

        if self.memory_duplicate_learning and str(self.episode.environment.controller.state) in self.episode.states:
            optimal_action = self.environment.controller.get_optimal_action(self.episode.target_object)
            self.duplicate_states_actions[self.action_num-1] = optimal_action

        if self.verbose:
            print(self.episode.actions_list[action])
        self.probs.append(prob)
        self.episode.action_probs.append(prob)
        self.entropies.append(entropy)
        self.values.append(out.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.episode.actions_record.append(action)
        if ('LayoutModel' in self.model_name) and self.model_phase == 'test':  # may training slow
            kl = self.model.object_distribution.kl
            self.episode.kl_record.append(kl)
        self.episode.prev_frame = model_input.state
        self.episode.current_frame = self.state()

        if self.learned_loss:
            res = torch.cat((self.hidden[0], self.last_action_probs), dim=1)
            if self.learned_input is None:
                self.learned_input = res
            else:
                self.learned_input = torch.cat((self.learned_input, res), dim=0)

        self._increment_episode_length()

        if self.episode.strict_done and action == DONE_ACTION_INT:
            self.success = self.info
            self.done = True
        elif self.done:
            self.success = not self.max_length

        return out.value, prob, action

    def reset_hidden(self, volatile=False):
        """ Reset the hidden state of the LSTM. """
        raise NotImplementedError()

    def repackage_hidden(self, volatile=False):
        """ Repackage the hidden state of the LSTM. """
        raise NotImplementedError()

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.probs = []
        self.reward = 0
        self.backprop_t = 0
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.learned_input = None
        self.learned_t = 0
        self.il_update_actions = {}
        self.action_num = 0
        self.meta_learning_actions = {}
        self.meta_predictions = []
        self.duplicate_states_actions = {}
        return self

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        raise NotImplementedError()

    def exit(self):
        """ Called on exit. """
        pass

    def reset_episode(self):
        """ Reset the episode so that it is identical. """
        return self._episode.reset()
