import torch
import numpy as np
import h5py
import torch.nn.functional as F
from utils.model_util import gpuify, toFloatTensor
from models.model_io import ModelInput
from reliability.gt_visibility_oracle import GTVisibilityOracle
from reliability.rule_lcr import RuleLocalContextReliability
from reliability.learned_lcr import LearnedLocalContextReliability
# from models.biasmodel import BiasModel
import time
import os
import hashlib

from .agent import ThorAgent


class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, scenes, targets, gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(NavigationAgent, self).__init__(
            create_model(args), args, rank, scenes, targets, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz
        self.keep_ori_obs = args.keep_ori_obs

        # time record
        self.time_start = torch.cuda.Event(enable_timing=True)
        self.time_end = torch.cuda.Event(enable_timing=True)

        self.glove = {}
        self.TDE_self = args.TDE_self
    
        if 'SP' in self.model_name:
            with h5py.File('/home/dhm/Code/vn/glove_map300d.hdf5', 'r') as rf:
                for i in rf:
                    self.glove[i] = rf[i][:]

    def eval_at_state(self, model_options):
        model_input = ModelInput()

        # model inputs
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        model_input.hidden = self.hidden

        current_detection_feature = self.episode.current_detection_feature()
        current_detection_feature = current_detection_feature[self.targets_index, :]
        target_embedding_array = np.zeros((len(self.targets), 1))
        target_embedding_array[self.targets.index(self.episode.target_object)] = 1

        self.episode.detection_results.append(
            list(current_detection_feature[self.targets.index(self.episode.target_object), 512:]))
        # if self.episode.current_cls_masks() is None:
        #     current_cls_masks = np.zeros((22,7,7))
        # else:
        #     current_cls_masks = self.episode.current_cls_masks()[()]

        target_embedding = {'appear': current_detection_feature[:, :512],
                            'info': current_detection_feature[:, 512:],
                            'indicator': target_embedding_array,
                            }
                            # 'masks':current_cls_masks}
        target_embedding['appear'] = toFloatTensor(target_embedding['appear'], self.gpu_id)
        target_embedding['info'] = toFloatTensor(target_embedding['info'], self.gpu_id)
        target_embedding['indicator'] = toFloatTensor(target_embedding['indicator'], self.gpu_id)
        # target_embedding['masks'] = toFloatTensor(target_embedding['masks'], self.gpu_id)
        model_input.target_class_embedding = target_embedding

        model_input.action_probs = self.last_action_probs

        model_input.scene = self.episode.environment.scene_name
        model_input.target_object = self.episode.target_object

        # -------------------------------------------------
        # Diagnostic trajectory-level reliability gate.
        #
        # Environment variable unset:
        #     exact original AKGVP behavior.
        #
        # Example:
        #     AKGVP_AUTO_GATE_RHO=0.1
        # -------------------------------------------------
        gate_rho = os.environ.get(
            "AKGVP_AUTO_GATE_RHO"
        )

        gate_mode = os.environ.get(
            "AKGVP_GATE_MODE",
            "strongest"
        )

        # -------------------------------------------------
        # Temporal diagnostic:
        # keep original semantic guidance during the
        # early search stage and activate attenuation
        # only after a fixed number of executed actions.
        #
        # Default 0 reproduces the existing gate exactly.
        # -------------------------------------------------
        gate_start_step = int(
            os.environ.get(
                "AKGVP_GATE_START_STEP",
                "0"
            )
        )

        gate_end_step_env = os.environ.get(
            "AKGVP_GATE_END_STEP"
        )

        gate_end_step = (
            None
            if gate_end_step_env is None
            else int(gate_end_step_env)
        )

        current_gate_step = len(
            self.episode.actions_record
        )

        gate_active = (
            current_gate_step
            >= gate_start_step
            and (
                gate_end_step is None
                or current_gate_step
                < gate_end_step
            )
        )

        if (
            gate_rho is not None
            and gate_active
        ):

            # ---------------------------------------------
            # Strongest semantic category:
            # existing trajectory intervention.
            # ---------------------------------------------
            if gate_mode == "strongest":
                model_input.auto_reliability_rho = (
                    float(gate_rho)
                )

            # ---------------------------------------------
            # Deterministic pseudo-random control.
            #
            # Select one currently detected non-goal
            # category, independent of AKGVP attention.
            # ---------------------------------------------
            elif gate_mode == "random":

                goal_idx = self.targets.index(
                    self.episode.target_object
                )

                det_scores = np.asarray(
                    current_detection_feature[:, -1]
                )

                valid_indices = np.flatnonzero(
                    det_scores > 0
                ).tolist()

                valid_indices = [
                    int(i)
                    for i in valid_indices
                    if int(i) != goal_idx
                ]

                if len(valid_indices) > 0:
                    state_key = str(
                        self.episode.environment
                        .controller.state
                    )

                    token = "{}|{}|{}".format(
                        self.episode.environment.scene_name,
                        state_key,
                        self.episode.target_object,
                    )

                    digest = hashlib.sha256(
                        token.encode("utf-8")
                    ).digest()

                    pseudo_random_value = int.from_bytes(
                        digest[:8],
                        byteorder="big",
                        signed=False,
                    )

                    gate_idx = valid_indices[
                        pseudo_random_value
                        % len(valid_indices)
                    ]

                    rho = torch.ones(
                        (
                            len(self.targets),
                            1,
                        ),
                        dtype=target_embedding[
                            "indicator"
                        ].dtype,
                        device=target_embedding[
                            "indicator"
                        ].device,
                    )

                    rho[
                        gate_idx, 0
                    ] = float(gate_rho)

                    model_input.reliability_weight = rho

                    self._last_random_gate_idx = (
                        gate_idx
                    )

                    if (
                        os.environ.get(
                            "AKGVP_GATE_DEBUG"
                        ) == "1"
                        and not hasattr(
                            self,
                            "_random_gate_debug_printed"
                        )
                    ):
                        print(
                            "[RANDOM_GATE DEBUG]",
                            "rho=",
                            gate_rho,
                            "idx=",
                            gate_idx,
                            "category=",
                            self.targets[gate_idx],
                            "scene=",
                            self.episode.environment.scene_name,
                            "target=",
                            self.episode.target_object,
                            flush=True,
                        )

                        self._random_gate_debug_printed = True

            # ---------------------------------------------
            # Uniform attenuation control.
            #
            # Suppress ALL non-goal semantic categories
            # by the same rho. This tests whether the
            # improvement comes simply from weakening
            # contextual semantic guidance.
            # ---------------------------------------------
            elif gate_mode == "uniform":

                goal_idx = self.targets.index(
                    self.episode.target_object
                )

                rho = torch.full(
                    (
                        len(self.targets),
                        1,
                    ),
                    float(gate_rho),
                    dtype=target_embedding[
                        "indicator"
                    ].dtype,
                    device=target_embedding[
                        "indicator"
                    ].device,
                )

                # Preserve the target category itself.
                rho[
                    goal_idx, 0
                ] = 1.0

                model_input.reliability_weight = rho


            # ---------------------------------------------
            # Episode-fixed random control.
            #
            # Select one detected non-goal category once
            # per episode and keep suppressing that same
            # category for the rest of the episode.
            #
            # This separates selective suppression from
            # per-step random/dropout-like perturbation.
            # ---------------------------------------------
            elif gate_mode == "episode_random":

                goal_idx = self.targets.index(
                    self.episode.target_object
                )

                # Select the category at the first state
                # where at least one non-goal detection
                # is available.
                if not hasattr(
                    self,
                    "_episode_random_gate_idx"
                ):
                    det_scores = np.asarray(
                        current_detection_feature[:, -1]
                    )

                    valid_indices = np.flatnonzero(
                        det_scores > 0
                    ).tolist()

                    valid_indices = [
                        int(i)
                        for i in valid_indices
                        if int(i) != goal_idx
                    ]

                    if len(valid_indices) > 0:
                        state_key = str(
                            self.episode.environment
                            .controller.state
                        )

                        token = "{}|{}|{}".format(
                            self.episode.environment.scene_name,
                            state_key,
                            self.episode.target_object,
                        )

                        digest = hashlib.sha256(
                            token.encode("utf-8")
                        ).digest()

                        value = int.from_bytes(
                            digest[:8],
                            byteorder="big",
                            signed=False,
                        )

                        self._episode_random_gate_idx = (
                            valid_indices[
                                value
                                % len(valid_indices)
                            ]
                        )

                if hasattr(
                    self,
                    "_episode_random_gate_idx"
                ):
                    gate_idx = int(
                        self._episode_random_gate_idx
                    )

                    rho = torch.ones(
                        (
                            len(self.targets),
                            1,
                        ),
                        dtype=target_embedding[
                            "indicator"
                        ].dtype,
                        device=target_embedding[
                            "indicator"
                        ].device,
                    )

                    rho[
                        gate_idx, 0
                    ] = float(gate_rho)

                    model_input.reliability_weight = rho


            # ---------------------------------------------
            # Privileged GT visibility oracle.
            #
            # Diagnostic / upper-bound experiment only.
            # Simulator GT is forbidden in the final method.
            # ---------------------------------------------
            elif gate_mode == "gt_visibility":

                if not hasattr(
                    self,
                    "_gt_visibility_oracle"
                ):
                    self._gt_visibility_oracle = (
                        GTVisibilityOracle(
                            classes=self.targets
                        )
                    )

                state_key = str(
                    self.episode.environment
                    .controller.state
                )

                det_scores = np.asarray(
                    current_detection_feature[
                        :, -1
                    ]
                )

                reliability_weight, gt_stats = (
                    self._gt_visibility_oracle
                    .build_weight(
                        scene=(
                            self.episode
                            .environment
                            .scene_name
                        ),
                        state=state_key,
                        target_name=(
                            self.episode
                            .target_object
                        ),
                        det_scores=det_scores,
                        rho=float(gate_rho),
                        device=(
                            target_embedding[
                                "indicator"
                            ].device
                        ),
                        dtype=(
                            target_embedding[
                                "indicator"
                            ].dtype
                        ),
                    )
                )

                model_input.reliability_weight = (
                    reliability_weight
                )

                self._last_gt_visibility_stats = (
                    gt_stats
                )

                if (
                    os.environ.get(
                        "AKGVP_GATE_DEBUG"
                    ) == "1"
                    and not hasattr(
                        self,
                        "_gt_visibility_debug_printed"
                    )
                ):
                    print(
                        "[GT_VISIBILITY DEBUG]",
                        "scene=",
                        self.episode.environment.scene_name,
                        "target=",
                        self.episode.target_object,
                        "state=",
                        state_key,
                        "suppressed=",
                        gt_stats["suppressed"],
                        "visible=",
                        gt_stats[
                            "confirmed_visible"
                        ],
                        flush=True,
                    )

                    self._gt_visibility_debug_printed = True


            # ---------------------------------------------
            # Scale ALL semantic categories equally.
            #
            # Diagnostic control:
            # goal and non-goal rows receive the same rho.
            # This separates overall semantic-branch
            # attenuation from target/context rebalancing.
            # ---------------------------------------------
            elif gate_mode == "all_semantic":

                reliability_weight = torch.full(
                    (
                        len(self.targets),
                        1,
                    ),
                    float(gate_rho),
                    dtype=target_embedding[
                        "indicator"
                    ].dtype,
                    device=target_embedding[
                        "indicator"
                    ].device,
                )

                model_input.reliability_weight = (
                    reliability_weight
                )


            elif gate_mode == "graph_branch":

                # Correct branch-level control:
                # preserve attention composition and
                # scale only the post-encoder graph feature.
                model_input.graph_branch_scale = (
                    float(gate_rho)
                )


            else:
                raise ValueError(
                    "Unknown AKGVP_GATE_MODE: {}".format(
                        gate_mode
                    )
                )

        # -------------------------------------------------
        # Local Context Reliability (LCR).
        #
        # off:
        #     exact original AKGVP / existing controls.
        #
        # rule:
        #     local negative-evidence reliability
        #     calibration.
        #
        # LCR and the old diagnostic gates are mutually
        # exclusive.
        # -------------------------------------------------
        lcr_mode = os.environ.get(
            "AKGVP_LCR_MODE",
            "off",
        ).strip().lower()

        if lcr_mode == "rule":

            if gate_rho is not None:
                raise RuntimeError(
                    "AKGVP_LCR_MODE=rule cannot be "
                    "combined with AKGVP_AUTO_GATE_RHO"
                )

            if not hasattr(
                self,
                "_rule_lcr",
            ):
                self._rule_lcr = (
                    RuleLocalContextReliability(
                        object_distribution=(
                            self.model
                            .object_distribution
                        ),
                        num_classes=(
                            len(self.targets)
                        ),
                    )
                )

            goal_idx = self.targets.index(
                self.episode.target_object
            )

            det_scores = np.asarray(
                current_detection_feature[
                    :,
                    -1,
                ],
                dtype=np.float32,
            )

            state_key = str(
                self.episode
                .environment
                .controller
                .state
            )

            reliability_np, lcr_stats = (
                self._rule_lcr.build_weight(
                    scene=(
                        self.episode
                        .environment
                        .scene_name
                    ),
                    state=state_key,
                    goal_idx=goal_idx,
                    detector_scores=(
                        det_scores
                    ),
                )
            )

            reliability_weight = (
                torch.as_tensor(
                    reliability_np,
                    dtype=(
                        target_embedding[
                            "indicator"
                        ].dtype
                    ),
                    device=(
                        target_embedding[
                            "indicator"
                        ].device
                    ),
                )
                .reshape(
                    -1,
                    1,
                )
            )

            model_input.reliability_weight = (
                reliability_weight
            )

            self._last_lcr_stats = (
                lcr_stats
            )

            if (
                os.environ.get(
                    "AKGVP_LCR_DEBUG"
                )
                == "1"
            ):
                print(
                    "[LCR]",
                    "step=",
                    len(
                        self.episode
                        .actions_record
                    ),
                    "scene=",
                    self.episode
                    .environment
                    .scene_name,
                    "target=",
                    self.episode
                    .target_object,
                    "region=",
                    lcr_stats[
                        "region_key"
                    ],
                    "new_view=",
                    lcr_stats[
                        "new_viewpoint"
                    ],
                    "views=",
                    lcr_stats[
                        "num_views"
                    ],
                    "support=",
                    round(
                        lcr_stats[
                            "semantic_support"
                        ],
                        4,
                    ),
                    "F=",
                    round(
                        lcr_stats[
                            "negative_evidence"
                        ],
                        4,
                    ),
                    "R=",
                    round(
                        lcr_stats[
                            "region_reliability"
                        ],
                        4,
                    ),
                    "gate_min=",
                    round(
                        lcr_stats[
                            "gate_min"
                        ],
                        4,
                    ),
                    "goal_gate=",
                    round(
                        lcr_stats[
                            "goal_gate"
                        ],
                        4,
                    ),
                    flush=True,
                )

        elif lcr_mode in ("learned", "learned_global", "constant", "constant_global", "selective", "action_aware"):

            if gate_rho is not None:
                raise RuntimeError(
                    "AKGVP_LCR_MODE=learned cannot be "
                    "combined with AKGVP_AUTO_GATE_RHO"
                )

            if not hasattr(
                self,
                "_learned_lcr",
            ):
                checkpoint_path = (
                    os.environ.get(
                        "AKGVP_LCR_CHECKPOINT",
                        (
                            "results/lcr_frozen/"
                            "learned_lcr_mlp.pt"
                        ),
                    )
                )

                self._learned_lcr = (
                    LearnedLocalContextReliability(
                        object_distribution=(
                            self.model
                            .object_distribution
                        ),
                        num_classes=(
                            len(self.targets)
                        ),
                        checkpoint_path=(
                            checkpoint_path
                        ),
                        min_views=3,
                        cell_size=1.0,
                        position_bin=0.5,
                        yaw_bin=90.0,
                        horizon_bin=30.0,
                        r_min=0.25,
                        constant_q=(
                            float(
                                os.environ[
                                    "AKGVP_LCR_CONSTANT_Q"
                                ]
                            )
                            if lcr_mode in (
                                "constant",
                                "constant_global",
                                "selective",
                                "action_aware",
                            )
                            else None
                        ),
                    )
                )

            if lcr_mode == "selective":

                self._learned_lcr.conflict_tau = float(
                    os.environ.get(
                        "AKGVP_LCR_CONFLICT_TAU",
                        "0.5",
                    )
                )

                self._learned_lcr.min_views = int(
                    os.environ.get(
                        "AKGVP_LCR_MIN_VIEWS",
                        "3",
                    )
                )

            else:

                # Preserve previous LCR modes exactly.
                self._learned_lcr.conflict_tau = None

            goal_idx = self.targets.index(
                self.episode.target_object
            )

            det_scores = np.asarray(
                current_detection_feature[
                    :,
                    -1,
                ],
                dtype=np.float32,
            )

            state_key = str(
                self.episode
                .environment
                .controller
                .state
            )

            reliability_np, lcr_stats = (
                self._learned_lcr
                .build_weight(
                    scene=(
                        self.episode
                        .environment
                        .scene_name
                    ),
                    state=state_key,
                    goal_idx=goal_idx,
                    detector_scores=(
                        det_scores
                    ),
                )
            )

            if (
                lcr_mode == "selective"
                and os.environ.get(
                    "AKGVP_LCR_DEBUG",
                    "0",
                ) == "1"
            ):
                print(
                    "[Selective-LCR]",
                    "views=",
                    lcr_stats.get(
                        "num_views",
                        -1,
                    ),
                    "conflict=",
                    round(
                        lcr_stats[
                            "conflict_score"
                        ],
                        4,
                    ),
                    "tau=",
                    lcr_stats[
                        "conflict_tau"
                    ],
                    "target_ev=",
                    round(
                        lcr_stats[
                            "target_evidence"
                        ],
                        4,
                    ),
                    "context=",
                    round(
                        lcr_stats[
                            "context_support"
                        ],
                        4,
                    ),
                    "activated=",
                    lcr_stats[
                        "activated"
                    ],
                    flush=True,
                )


            # -------------------------------------------------
            # Ablation: learned reliability with uniform
            # context calibration.
            #
            # Everything up to R is identical to Learned-LCR.
            # Only relation-specific attribution is removed:
            #   gate[c] = R, c != goal
            #   gate[goal] = 1
            # -------------------------------------------------
            if lcr_mode in (
                "learned_global",
                "constant_global",
            ):

                current_r = float(
                    lcr_stats["R_before"]
                )

                reliability_np = np.full(
                    len(self.targets),
                    current_r,
                    dtype=np.float32,
                )

                reliability_np[
                    goal_idx
                ] = 1.0

                lcr_stats = dict(
                    lcr_stats
                )

                lcr_stats[
                    "gate_min"
                ] = float(
                    reliability_np.min()
                )

                lcr_stats[
                    "goal_gate"
                ] = 1.0

                lcr_stats[
                    "calibration_mode"
                ] = "global"

            reliability_weight = (
                torch.as_tensor(
                    reliability_np,
                    dtype=(
                        target_embedding[
                            "indicator"
                        ].dtype
                    ),
                    device=(
                        target_embedding[
                            "indicator"
                        ].device
                    ),
                )
                .reshape(
                    -1,
                    1,
                )
            )

            if lcr_mode == "action_aware":

                # Save the candidate LCR gate.
                # The first forward must remain exact Base.
                self._action_aware_lcr_weight = (
                    reliability_weight
                )

                model_input.reliability_weight = (
                    None
                )

            else:

                model_input.reliability_weight = (
                    reliability_weight
                )

            self._last_lcr_stats = (
                lcr_stats
            )

        elif lcr_mode not in (
            "",
            "off",
            "none",
        ):
            raise ValueError(
                "Unknown AKGVP_LCR_MODE: "
                "{}".format(
                    lcr_mode
                )
            )


        if 'Memory' in self.model_name:
            state_length = self.hidden_state_sz

            if len(self.episode.state_reps) == 0:
                model_input.states_rep = torch.zeros(1, state_length)
            else:
                model_input.states_rep = torch.stack(self.episode.state_reps)

            dim_obs = 512
            if len(self.episode.obs_reps) == 0:
                model_input.obs_reps = torch.zeros(1, dim_obs)
            else:
                model_input.obs_reps = torch.stack(self.episode.obs_reps)

            if len(self.episode.state_memory) == 0:
                model_input.states_memory = torch.zeros(1, state_length)
            else:
                model_input.states_memory = torch.stack(self.episode.state_memory)

            if len(self.episode.action_memory) == 0:
                model_input.action_memory = torch.zeros(1, 6)
            else:
                model_input.action_memory = torch.stack(self.episode.action_memory)

            model_input.states_rep = toFloatTensor(model_input.states_rep, self.gpu_id)
            model_input.states_memory = toFloatTensor(model_input.states_memory, self.gpu_id)
            model_input.action_memory = toFloatTensor(model_input.action_memory, self.gpu_id)
            model_input.obs_reps = toFloatTensor(model_input.obs_reps, self.gpu_id)

        if self.TDE_self:
            output = self.model.forward(model_input, model_options)
            output_counterfact = self.model.forward(model_input, model_options, counterfact=True)

            scale = self.model.object_distribution.kl - self.model.TDE_threshold
            # print(self.model.object_distribution.kl)
            scale_min = self.model.scale_min.to(scale.device)
            scale_max = self.model.scale_max.to(scale.device)
            scale = torch.min(scale_max, scale)
            scale = torch.max(scale_min, scale)  # same as reLu when scale_min=0.0
            # output.logit = output.logit - torch.mul(output_counterfact.logit, scale)
            output.logit = torch.mul(output.logit, 1.0+scale) - torch.mul(output_counterfact.logit, scale)
        else:

            if lcr_mode == "action_aware":

                if not hasattr(
                    self,
                    "_action_aware_lcr_weight",
                ):
                    raise RuntimeError(
                        "Missing Action-Aware LCR weight"
                    )

                # =========================================
                # Candidate 1: exact Base
                #
                # Only this forward updates AKGVP semantic
                # memory for the current observation.
                # =========================================
                model_input.reliability_weight = None
                model_input.skip_object_memory_update = (
                    False
                )

                output_base = self.model.forward(
                    model_input,
                    model_options,
                )

                # =========================================
                # Candidate 2: LCR
                #
                # Same observation, recurrent input and
                # already-updated semantic memory.
                # Do not observe the frame twice.
                # =========================================
                model_input.reliability_weight = (
                    self._action_aware_lcr_weight
                )

                model_input.skip_object_memory_update = (
                    True
                )

                try:
                    output_lcr = self.model.forward(
                        model_input,
                        model_options,
                    )
                finally:
                    model_input.skip_object_memory_update = (
                        False
                    )

                # =========================================
                # Action-level arbitration
                # =========================================
                prob_base = torch.softmax(
                    output_base.logit,
                    dim=1,
                )

                prob_lcr = torch.softmax(
                    output_lcr.logit,
                    dim=1,
                )

                top_base = torch.topk(
                    prob_base,
                    k=2,
                    dim=1,
                ).values

                top_lcr = torch.topk(
                    prob_lcr,
                    k=2,
                    dim=1,
                ).values

                margin_base = float(
                    (
                        top_base[0, 0]
                        - top_base[0, 1]
                    )
                    .detach()
                    .cpu()
                    .item()
                )

                margin_lcr = float(
                    (
                        top_lcr[0, 0]
                        - top_lcr[0, 1]
                    )
                    .detach()
                    .cpu()
                    .item()
                )

                delta_margin = (
                    margin_lcr
                    - margin_base
                )

                action_base = int(
                    prob_base.argmax(
                        dim=1
                    )[0].item()
                )

                action_lcr = int(
                    prob_lcr.argmax(
                        dim=1
                    )[0].item()
                )

                guard_tau = float(
                    os.environ.get(
                        "AKGVP_BASE_GUARD_TAU",
                        "0.05",
                    )
                )

                guard_mode = os.environ.get(
                    "AKGVP_GUARD_MODE",
                    "rule",
                ).strip().lower()

                # =========================================
                # Rule Guarded-LCR
                # =========================================
                rule_use_base = bool(
                    action_lcr != action_base
                    and
                    margin_base >= guard_tau
                )

                learned_guard_prob = None

                # =========================================
                # Arbitration
                #
                # rule:
                #   Original confidence Guarded-LCR.
                #
                # learned:
                #   When Base/LCR disagree, predict whether
                #   LCR should override Base using the
                #   frozen 28->1 Linear Intervention Guard.
                #
                #   y=1 means prefer LCR.
                #
                # When actions agree, retain the LCR branch
                # so its recurrent trajectory is preserved.
                # =========================================
                if guard_mode == "rule":

                    use_base = rule_use_base
                    use_lcr = not use_base

                elif guard_mode == "learned":

                    if action_base == action_lcr:

                        use_base = False
                        use_lcr = True

                    else:

                        # ---------------------------------
                        # Load frozen Linear Guard once per
                        # agent/process.
                        # ---------------------------------
                        if not hasattr(
                            self,
                            "_learned_guard_weight",
                        ):

                            guard_ckpt_path = (
                                os.environ.get(
                                    "AKGVP_LEARNED_GUARD_CHECKPOINT",
                                    (
                                        "results/"
                                        "learned_guard_protocol/"
                                        "linear_guard_final.pt"
                                    ),
                                )
                            )

                            guard_ckpt = torch.load(
                                guard_ckpt_path,
                                map_location="cpu",
                            )

                            state_dict = (
                                guard_ckpt[
                                    "state_dict"
                                ]
                            )

                            self._learned_guard_weight = (
                                state_dict[
                                    "weight"
                                ]
                                .detach()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                                .astype(
                                    np.float32
                                )
                            )

                            self._learned_guard_bias = float(
                                state_dict[
                                    "bias"
                                ]
                                .detach()
                                .cpu()
                                .reshape(-1)[0]
                                .item()
                            )

                            self._learned_guard_mean = (
                                np.asarray(
                                    guard_ckpt[
                                        "feature_mean"
                                    ],
                                    dtype=np.float32,
                                )
                                .reshape(-1)
                            )

                            self._learned_guard_std = (
                                np.asarray(
                                    guard_ckpt[
                                        "feature_std"
                                    ],
                                    dtype=np.float32,
                                )
                                .reshape(-1)
                            )

                            if (
                                self._learned_guard_weight.shape
                                != (28,)
                                or
                                self._learned_guard_mean.shape
                                != (28,)
                                or
                                self._learned_guard_std.shape
                                != (28,)
                            ):
                                raise RuntimeError(
                                    "Learned Guard checkpoint "
                                    "must contain 28-D features."
                                )

                        # ---------------------------------
                        # Same 28-D feature definition used
                        # during training.
                        # ---------------------------------
                        entropy_base_guard = float(
                            -(
                                prob_base
                                * torch.log(
                                    prob_base.clamp_min(
                                        1e-8
                                    )
                                )
                            )
                            .sum()
                            .detach()
                            .cpu()
                            .item()
                        )

                        entropy_lcr_guard = float(
                            -(
                                prob_lcr
                                * torch.log(
                                    prob_lcr.clamp_min(
                                        1e-8
                                    )
                                )
                            )
                            .sum()
                            .detach()
                            .cpu()
                            .item()
                        )

                        guard_stats = dict(
                            getattr(
                                self,
                                "_last_lcr_stats",
                                {},
                            )
                        )

                        def _guard_scalar(name):

                            value = guard_stats.get(
                                name,
                                None,
                            )

                            if value is None:
                                return 0.0

                            if torch.is_tensor(
                                value
                            ):
                                value = (
                                    value
                                    .detach()
                                    .reshape(-1)[0]
                                    .cpu()
                                    .item()
                                )

                            if isinstance(
                                value,
                                np.generic,
                            ):
                                value = value.item()

                            try:
                                return float(value)
                            except Exception:
                                return 0.0

                        pb_guard = (
                            prob_base
                            .detach()
                            .reshape(-1)
                            .cpu()
                            .numpy()
                            .astype(
                                np.float32
                            )
                        )

                        pl_guard = (
                            prob_lcr
                            .detach()
                            .reshape(-1)
                            .cpu()
                            .numpy()
                            .astype(
                                np.float32
                            )
                        )

                        feature_guard = np.concatenate(
                            [
                                pb_guard,
                                pl_guard,
                                pl_guard - pb_guard,

                                np.asarray(
                                    [
                                        margin_base,
                                        margin_lcr,

                                        entropy_base_guard,
                                        entropy_lcr_guard,

                                        _guard_scalar(
                                            "target_evidence"
                                        ),

                                        _guard_scalar(
                                            "context_support"
                                        ),

                                        _guard_scalar(
                                            "conflict_score"
                                        ),

                                        _guard_scalar(
                                            "num_views"
                                        ),

                                        _guard_scalar(
                                            "R_before"
                                        ),

                                        _guard_scalar(
                                            "gate_min"
                                        ),
                                    ],
                                    dtype=np.float32,
                                ),
                            ]
                        )

                        if feature_guard.shape != (28,):
                            raise RuntimeError(
                                "Learned Guard feature "
                                "dimension != 28"
                            )

                        feature_guard = (
                            (
                                feature_guard
                                - self._learned_guard_mean
                            )
                            /
                            self._learned_guard_std
                        )

                        learned_guard_logit = float(
                            np.dot(
                                self._learned_guard_weight,
                                feature_guard,
                            )
                            + self._learned_guard_bias
                        )

                        # Numerically stable sigmoid.
                        if learned_guard_logit >= 0.0:
                            learned_guard_prob = (
                                1.0
                                /
                                (
                                    1.0
                                    + np.exp(
                                        -learned_guard_logit
                                    )
                                )
                            )
                        else:
                            ez = np.exp(
                                learned_guard_logit
                            )
                            learned_guard_prob = (
                                ez
                                / (1.0 + ez)
                            )

                        learned_guard_tau = float(
                            os.environ.get(
                                "AKGVP_LEARNED_GUARD_TAU",
                                "0.5",
                            )
                        )

                        use_lcr = bool(
                            learned_guard_prob
                            >= learned_guard_tau
                        )

                        use_base = not use_lcr

                        if (
                            os.environ.get(
                                "AKGVP_LEARNED_GUARD_DEBUG",
                                "0",
                            )
                            == "1"
                        ):
                            print(
                                "[LEARNED-GUARD]",
                                "scene=",
                                self.episode
                                .environment
                                .scene_name,
                                "target=",
                                self.episode.target_object,
                                "step=",
                                len(
                                    self.episode
                                    .actions_record
                                ),
                                "base=",
                                action_base,
                                "lcr=",
                                action_lcr,
                                "p_lcr=",
                                round(
                                    learned_guard_prob,
                                    6,
                                ),
                                "tau=",
                                learned_guard_tau,
                                "use_lcr=",
                                use_lcr,
                                flush=True,
                            )

                else:

                    raise ValueError(
                        "Unknown AKGVP_GUARD_MODE="
                        + str(guard_mode)
                        + "; expected rule or learned."
                    )

                # =========================================
                # Optional oracle logger for training a
                # lightweight learned intervention guard.
                #
                # Disabled unless:
                # AKGVP_GUARD_SAMPLE_DIR is set.
                #
                # Logging never changes the selected action.
                # =========================================
                guard_sample_dir = os.environ.get(
                    "AKGVP_GUARD_SAMPLE_DIR",
                    "",
                ).strip()

                if (
                    guard_sample_dir
                    and action_base != action_lcr
                ):
                    import json as _json

                    try:
                        # ---------------------------------
                        # Training-only oracle supervision.
                        #
                        # The evaluation controller does not
                        # necessarily populate
                        # controller.optimal_action, so load
                        # optimal_action.json directly.
                        #
                        # Cache one scene at a time to avoid
                        # repeated disk I/O.
                        # ---------------------------------
                        controller = (
                            self.episode
                            .environment
                            .controller
                        )

                        oracle_scene = (
                            self.episode
                            .environment
                            .scene_name
                        )

                        if (
                            getattr(
                                self,
                                "_guard_oracle_scene",
                                None,
                            )
                            != oracle_scene
                        ):
                            oracle_root = getattr(
                                controller,
                                "offline_data_dir",
                                "datasets/Scene_Data",
                            )

                            oracle_name = getattr(
                                controller,
                                "optimal_action_file_name",
                                None,
                            )

                            if not oracle_name:
                                oracle_name = (
                                    "optimal_action.json"
                                )

                            oracle_path = os.path.join(
                                oracle_root,
                                oracle_scene,
                                oracle_name,
                            )

                            with open(
                                oracle_path,
                                "r",
                                encoding="utf-8",
                            ) as fp:
                                self._guard_oracle_data = (
                                    _json.load(fp)
                                )

                            self._guard_oracle_scene = (
                                oracle_scene
                            )

                        oracle_state = str(
                            controller.state
                        )

                        optimal_action = (
                            self._guard_oracle_data[
                                oracle_state
                            ][
                                self.episode
                                .target_object
                            ]
                        )

                        if torch.is_tensor(
                            optimal_action
                        ):
                            optimal_action = (
                                optimal_action
                                .detach()
                                .reshape(-1)[0]
                                .cpu()
                                .item()
                            )

                        if isinstance(
                            optimal_action,
                            np.generic,
                        ):
                            optimal_action = (
                                optimal_action.item()
                            )

                        if isinstance(
                            optimal_action,
                            str,
                        ):
                            try:
                                optimal_action = int(
                                    optimal_action
                                )
                            except ValueError:
                                optimal_action = (
                                    self.episode
                                    .actions_list
                                    .index(
                                        optimal_action
                                    )
                                )

                        optimal_action = int(
                            optimal_action
                        )

                    except Exception as exc:
                        print(
                            "[GUARD-ORACLE-ERROR]",
                            "scene=",
                            self.episode.environment.scene_name,
                            "target=",
                            self.episode.target_object,
                            "state=",
                            str(
                                self.episode
                                .environment
                                .controller
                                .state
                            ),
                            "type=",
                            type(exc).__name__,
                            "error=",
                            repr(exc),
                            flush=True,
                        )
                        optimal_action = None

                    if optimal_action is not None:

                        base_correct = bool(
                            action_base
                            == optimal_action
                        )

                        lcr_correct = bool(
                            action_lcr
                            == optimal_action
                        )

                        if (
                            lcr_correct
                            and not base_correct
                        ):
                            guard_label = 1

                        elif (
                            base_correct
                            and not lcr_correct
                        ):
                            guard_label = 0

                        else:
                            guard_label = None

                        entropy_base = float(
                            -(
                                prob_base
                                * torch.log(
                                    prob_base.clamp_min(
                                        1e-8
                                    )
                                )
                            )
                            .sum()
                            .detach()
                            .cpu()
                            .item()
                        )

                        entropy_lcr = float(
                            -(
                                prob_lcr
                                * torch.log(
                                    prob_lcr.clamp_min(
                                        1e-8
                                    )
                                )
                            )
                            .sum()
                            .detach()
                            .cpu()
                            .item()
                        )

                        local_stats = dict(
                            getattr(
                                self,
                                "_last_lcr_stats",
                                {},
                            )
                        )

                        def _scalar(name):
                            value = local_stats.get(
                                name,
                                None,
                            )

                            if value is None:
                                return None

                            if torch.is_tensor(
                                value
                            ):
                                value = (
                                    value
                                    .detach()
                                    .reshape(-1)[0]
                                    .cpu()
                                    .item()
                                )

                            if isinstance(
                                value,
                                np.generic,
                            ):
                                value = value.item()

                            try:
                                return float(value)
                            except Exception:
                                return None

                        sample = {
                            "scene": (
                                self.episode
                                .environment
                                .scene_name
                            ),

                            "target": (
                                self.episode
                                .target_object
                            ),

                            "state": str(
                                self.episode
                                .environment
                                .controller
                                .state
                            ),

                            "step": int(
                                len(
                                    self.episode
                                    .actions_record
                                )
                            ),

                            "prob_base": (
                                prob_base
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "prob_lcr": (
                                prob_lcr
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "action_base":
                                int(action_base),

                            "action_lcr":
                                int(action_lcr),

                            "optimal_action":
                                int(optimal_action),

                            "base_correct":
                                base_correct,

                            "lcr_correct":
                                lcr_correct,

                            "label":
                                guard_label,

                            "margin_base":
                                float(margin_base),

                            "margin_lcr":
                                float(margin_lcr),

                            "delta_margin":
                                float(delta_margin),

                            "entropy_base":
                                entropy_base,

                            "entropy_lcr":
                                entropy_lcr,

                            "target_evidence":
                                _scalar(
                                    "target_evidence"
                                ),

                            "context_support":
                                _scalar(
                                    "context_support"
                                ),

                            "conflict_score":
                                _scalar(
                                    "conflict_score"
                                ),

                            "num_views":
                                _scalar(
                                    "num_views"
                                ),

                            "R_before":
                                _scalar(
                                    "R_before"
                                ),

                            "gate_min":
                                _scalar(
                                    "gate_min"
                                ),

                            "rule_use_base":
                                bool(rule_use_base),

                            "rule_tau":
                                float(guard_tau),
                        }

                        os.makedirs(
                            guard_sample_dir,
                            exist_ok=True,
                        )

                        sample_path = os.path.join(
                            guard_sample_dir,
                            "guard_samples_{}.jsonl"
                            .format(
                                os.getpid()
                            ),
                        )

                        with open(
                            sample_path,
                            "a",
                            encoding="utf-8",
                        ) as fp:
                            fp.write(
                                _json.dumps(
                                    sample,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                output = (
                    output_lcr
                    if use_lcr
                    else output_base
                )

                # Keep diagnostics consistent with the
                # policy branch that is actually executed.
                model_input.reliability_weight = (
                    self._action_aware_lcr_weight
                    if use_lcr
                    else None
                )

                current_stats = dict(
                    getattr(
                        self,
                        "_last_lcr_stats",
                        {},
                    )
                )

                current_stats.update(
                    {
                        "arb_margin_base":
                            margin_base,

                        "arb_margin_lcr":
                            margin_lcr,

                        "arb_delta_margin":
                            delta_margin,

                        "arb_action_base":
                            action_base,

                        "arb_action_lcr":
                            action_lcr,

                        "arb_action_diff":
                            bool(
                                action_base
                                != action_lcr
                            ),

                        "arb_use_lcr":
                            use_lcr,

                        "arb_tau":
                            guard_tau,
                    }
                )

                self._last_lcr_stats = (
                    current_stats
                )

                # =========================================
                # Case-only semantic trace
                #
                # Logging only. No action, probability,
                # recurrent state, or arbitration variable
                # is modified here.
                # =========================================
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

                        def _case_scalar(_name):
                            _v = current_stats.get(
                                _name,
                                None,
                            )

                            if _v is None:
                                return None

                            try:
                                if hasattr(
                                    _v,
                                    "detach",
                                ):
                                    _v = (
                                        _v.detach()
                                        .cpu()
                                        .reshape(-1)[0]
                                        .item()
                                    )

                                elif isinstance(
                                    _v,
                                    (list, tuple),
                                ):
                                    if len(_v) == 0:
                                        return None
                                    _v = _v[0]

                                return float(_v)

                            except Exception:
                                return None

                        _case_semantic_event = {
                            "step": int(
                                len(
                                    self.episode
                                    .actions_record
                                )
                            ),

                            "state": str(
                                self.episode
                                .environment
                                .controller
                                .state
                            ),

                            "prob_base": (
                                prob_base
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "prob_calibrated": (
                                prob_lcr
                                .detach()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            ),

                            "action_base":
                                int(action_base),

                            "action_calibrated":
                                int(action_lcr),

                            "action_diff":
                                bool(
                                    action_base
                                    != action_lcr
                                ),

                            "selected_branch": (
                                "calibrated"
                                if use_lcr
                                else "base"
                            ),

                            "selected_action": int(
                                action_lcr
                                if use_lcr
                                else action_base
                            ),

                            "margin_base":
                                float(margin_base),

                            "margin_calibrated":
                                float(margin_lcr),

                            "delta_margin":
                                float(delta_margin),

                            "guard_tau":
                                float(guard_tau),

                            "target_evidence":
                                _case_scalar(
                                    "target_evidence"
                                ),

                            "context_support":
                                _case_scalar(
                                    "context_support"
                                ),

                            "conflict_score":
                                _case_scalar(
                                    "conflict_score"
                                ),

                            "num_views":
                                _case_scalar(
                                    "num_views"
                                ),

                            "R_before":
                                _case_scalar(
                                    "R_before"
                                ),

                            "gate_min":
                                _case_scalar(
                                    "gate_min"
                                ),
                        }

                        self.episode.case_semantic_trace.append(
                            _case_semantic_event
                        )

                # =========================================
                # Intervention-frequency diagnostics
                #
                # Logging only. This block must never
                # modify action selection or recurrent state.
                # =========================================
                if (
                    os.environ.get(
                        "AKGVP_FREQ_DIAG",
                        "0",
                    ).strip()
                    == "1"
                ):
                    self.episode.diag_arb_steps = (
                        int(
                            getattr(
                                self.episode,
                                "diag_arb_steps",
                                0,
                            )
                        )
                        + 1
                    )

                    if action_base != action_lcr:
                        self.episode.diag_arb_disagree = (
                            int(
                                getattr(
                                    self.episode,
                                    "diag_arb_disagree",
                                    0,
                                )
                            )
                            + 1
                        )

                        if use_base:
                            self.episode.diag_disagree_use_base = (
                                int(
                                    getattr(
                                        self.episode,
                                        "diag_disagree_use_base",
                                        0,
                                    )
                                )
                                + 1
                            )
                        else:
                            self.episode.diag_disagree_use_lcr = (
                                int(
                                    getattr(
                                        self.episode,
                                        "diag_disagree_use_lcr",
                                        0,
                                    )
                                )
                                + 1
                            )

                    if use_lcr:
                        self.episode.diag_lcr_selected = (
                            int(
                                getattr(
                                    self.episode,
                                    "diag_lcr_selected",
                                    0,
                                )
                            )
                            + 1
                        )

                if (
                    os.environ.get(
                        "AKGVP_ACTION_AWARE_DEBUG",
                        "0",
                    )
                    == "1"
                ):
                    print(
                        "[ACTION-AWARE]",
                        "base=",
                        action_base,
                        "lcr=",
                        action_lcr,
                        "mb=",
                        round(
                            margin_base,
                            4,
                        ),
                        "ml=",
                        round(
                            margin_lcr,
                            4,
                        ),
                        "delta=",
                        round(
                            delta_margin,
                            4,
                        ),
                        "tau=",
                        guard_tau,
                        "use_lcr=",
                        use_lcr,
                        flush=True,
                    )

            else:

                output = self.model.forward(
                    model_input,
                    model_options,
                )

        # -------------------------------------------------
        # Learned-LCR post-forward update.
        #
        # IMPORTANT:
        # Current base_attention / semantic KL are observed
        # only AFTER this forward. The updated reliability
        # therefore becomes active from the NEXT navigation
        # step, preventing a causal loop.
        # -------------------------------------------------
        if lcr_mode in ("learned", "learned_global", "constant", "constant_global", "selective", "action_aware"):

            base_attention = getattr(
                self.model,
                "last_attention_weight_base",
                None,
            )

            if base_attention is None:
                raise RuntimeError(
                    "Learned-LCR requires "
                    "last_attention_weight_base"
                )

            goal_idx = self.targets.index(
                self.episode.target_object
            )

            det_scores = np.asarray(
                current_detection_feature[
                    :,
                    -1,
                ],
                dtype=np.float32,
            )

            obj_dist = getattr(
                self.model,
                "object_distribution",
                None,
            )

            kl_value = getattr(
                obj_dist,
                "kl",
                0.0,
            )

            if torch.is_tensor(
                kl_value
            ):
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

            state_key = str(
                self.episode
                .environment
                .controller
                .state
            )

            post_stats = (
                self._learned_lcr
                .observe_after_forward(
                    state=state_key,
                    goal_idx=goal_idx,
                    detector_scores=(
                        det_scores
                    ),
                    base_attention=(
                        base_attention
                        .detach()
                        .reshape(-1)
                        .cpu()
                        .numpy()
                    ),
                    semantic_kl=(
                        semantic_kl
                    ),
                )
            )

            # Preserve gate information from the actual
            # current forward and append the post-forward
            # predictor update.
            current_stats = dict(
                getattr(
                    self,
                    "_last_lcr_stats",
                    {},
                )
            )

            current_stats.update(
                post_stats
            )

            self._last_lcr_stats = (
                current_stats
            )

            if (
                os.environ.get(
                    "AKGVP_LCR_DEBUG"
                )
                == "1"
            ):
                print(
                    "[LCR-LEARNED]",
                    "step=",
                    len(
                        self.episode
                        .actions_record
                    ),
                    "scene=",
                    self.episode
                    .environment
                    .scene_name,
                    "target=",
                    self.episode
                    .target_object,
                    "region=",
                    current_stats[
                        "region_key"
                    ],
                    "new_view=",
                    current_stats[
                        "new_viewpoint"
                    ],
                    "views_before=",
                    current_stats[
                        "num_views_before"
                    ],
                    "views=",
                    current_stats[
                        "num_views"
                    ],
                    "q=",
                    (
                        None
                        if current_stats[
                            "q"
                        ] is None
                        else round(
                            current_stats[
                                "q"
                            ],
                            4,
                        )
                    ),
                    "R_before=",
                    round(
                        current_stats[
                            "R_before"
                        ],
                        4,
                    ),
                    "R_after=",
                    round(
                        current_stats[
                            "R_after"
                        ],
                        4,
                    ),
                    "gate_min=",
                    round(
                        current_stats[
                            "gate_min"
                        ],
                        4,
                    ),
                    "goal_gate=",
                    round(
                        current_stats[
                            "goal_gate"
                        ],
                        4,
                    ),
                    flush=True,
                )

        # Optional one-time diagnostic.
        if (
            os.environ.get(
                "AKGVP_AUTO_GATE_DEBUG"
            ) == "1"
            and not hasattr(
                self,
                "_auto_gate_debug_printed"
            )
        ):
            print(
                "[AUTO_GATE DEBUG]",
                "input_rho=",
                gate_rho,
                "scene=",
                self.episode.environment.scene_name,
                "target=",
                self.episode.target_object,
                "selected_idx=",
                getattr(
                    self.model,
                    "last_reliability_gate_idx",
                    None,
                ),
                "applied_rho=",
                getattr(
                    self.model,
                    "last_reliability_gate_rho",
                    None,
                ),
                flush=True,
            )

            self._auto_gate_debug_printed = True

        return model_input, output

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):

        # New episode -> clear episode-fixed
        # random semantic control.
        if hasattr(
            self,
            "_episode_random_gate_idx"
        ):
            del self._episode_random_gate_idx

        # New episode -> clear local reliability memory.
        if hasattr(
            self,
            "_rule_lcr",
        ):
            self._rule_lcr.reset()


        if hasattr(
            self,
            "_learned_lcr",
        ):
            self._learned_lcr.reset()

        self._last_lcr_stats = None


        with torch.cuda.device(self.gpu_id):
            self.hidden = (
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
            )

        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

        if hasattr(self.model, 'reset'):
            self.model.reset()
        if hasattr(self.model, 'object_distribution'):
            self.model.object_distribution.reset_memory()

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass
