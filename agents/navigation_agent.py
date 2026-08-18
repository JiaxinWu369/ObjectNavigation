import torch
import numpy as np
import h5py
import torch.nn.functional as F
from utils.model_util import gpuify, toFloatTensor
from models.model_io import ModelInput
from reliability.gt_visibility_oracle import GTVisibilityOracle
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
            output = self.model.forward(model_input, model_options)

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
