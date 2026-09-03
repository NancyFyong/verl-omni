# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher score provider contract for the DMD inner loop.

The existing ``DiffusionTeacherManager.compute_prev_sample_mean()`` returns a
single transition output on CPU and is optimized for post-rollout OPD. It is not
suitable for the DMD inner loop, which repeatedly scores newly generated GPU
latents at arbitrary score sigmas. This module defines a narrower provider
contract (RFC §13.2). PR 1 defines the protocol; ``ColocatedTeacherScoreProvider``
and ``RayTeacherScoreProvider`` wire in with the role runtime in PR 2+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from verl_omni.trainer.diffusion.distillation.contracts import CanonicalPrediction, ScoreBatch

if TYPE_CHECKING:
    pass

__all__ = ["TeacherScoreProvider"]


@runtime_checkable
class TeacherScoreProvider(Protocol):
    """Provides canonical ``x0`` teacher predictions for a score batch."""

    def predict_x0(self, score_batch: ScoreBatch) -> CanonicalPrediction:
        """Return the teacher's canonical fp32 ``x0`` for ``score_batch``.

        This must not hold a student autograd graph across an unbounded
        synchronous round trip. A standalone provider is responsible for
        versioned weight sync, cancellation, and never materializing tensors on
        the driver via the driver process.
        """
        ...
