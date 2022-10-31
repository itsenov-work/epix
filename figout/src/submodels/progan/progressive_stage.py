from framework.training_stage import TransitionTrainingStage
from framework.submodel import Submodel, Block

from utils.layer_utilities import DuplicatingLayer, OnOffButton, Alpha, AlphaMergerLayer
from abc import ABC


class MergerButton(OnOffButton):
    def off(self, inputs):
        input1, input2 = inputs
        return input1


class ProgressiveStage(ABC, TransitionTrainingStage):
    def __init__(self,
                 adversary: Submodel,
                 scaling_block: Block = None,
                 temporary_block: Block = None,
                 # TODO: this is not injected yet
                 transition_steps: int = 50
                 ):
        super(ProgressiveStage, self).__init__()
        self.transition_steps = transition_steps
        self.main_block = adversary
        self.scaling_block = scaling_block if scaling_block is not None else Block()
        self.temporary_block = temporary_block if temporary_block is not None else Block()

        self.splitter = DuplicatingLayer()
        self.alpha = Alpha(self.transition_steps)
        self.alpha_merger = MergerButton(AlphaMergerLayer(self.alpha), self.in_transition)
        self.finalize()

    def finalize(self):
        self.main_block.initial_block = OnOffButton(self.main_block.initial_block, self.is_initial)
        self.main_block.finalizing_block = OnOffButton(self.main_block.finalizing_block, self.is_final)
        self.main_block = OnOffButton(self.main_block, self.active)

        if self.scaling_block:
            self.scaling_block = OnOffButton(self.scaling_block, self.active)

        if self.temporary_block:
            self.temporary_block = OnOffButton(self.temporary_block, self.in_transition)
