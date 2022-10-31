from framework.submodel import Block
from submodels.progan.progressive_adversary import ProgressiveAdversary
from submodels.progan.progressive_stage import ProgressiveStage
from utils.layer_utilities import Downscale2D


class ProgressiveCritic(ProgressiveAdversary):
    def __init__(self, *args, **kwargs):
        super(ProgressiveCritic, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        for i, stage in enumerate(self.stages[:0:-1]):
            inputs_main, inputs_residual = stage.splitter(inputs)
            inputs_main = stage.main_block(inputs_main)
            inputs_main = stage.scaling_block(inputs_main)
            inputs_residual = stage.scaling_block(inputs_residual)
            inputs_residual = stage.temporary_block(inputs_residual)
            inputs = stage.alpha_merger([inputs_main, inputs_residual])

        inputs = self.stages[0].main_block(inputs)
        return inputs

    def create_stage(self, index: int):
        main_block = self.adversaries[index]
        temporary_block = Block()
        scaling_block = None
        if not main_block.initial_block:
            main_block.initial_block.append(main_block.coreSegment(main_block.filters[-1], kernel_size=(1, 1)))
            temporary_block.extend(main_block.coreSegment(main_block.filters[-1], kernel_size=(1, 1)))
        else:
            temporary_block.extend(main_block.initialLayers())
        if index != 0:
            scaling_block = Downscale2D(factor=self.factor_from_resolutions(
                self.resolutions[index - 1], self.resolutions[index]
            ))

        return ProgressiveStage(adversary=main_block, scaling_block=scaling_block, temporary_block=temporary_block)
