from framework.submodel import Block
from submodels.progan.progressive_adversary import ProgressiveAdversary
from submodels.progan.progressive_stage import ProgressiveStage
from utils.layer_utilities import Upscale2D


class ProgressiveArtist(ProgressiveAdversary):
    def call(self, inputs, **kwargs):
        inputs = self.stages[0].main_block(inputs)
        for stage in self.stages[1:]:
            inputs = stage.scaling_block(inputs)
            inputs_main, inputs_residual = stage.splitter(inputs)
            inputs_main = stage.main_block(inputs_main)
            inputs_residual = stage.temporary_block(inputs_residual)
            inputs = stage.alpha_merger([inputs_main, inputs_residual])

        return inputs

    def create_stage(self, index: int):
        main_block = self.adversaries[index]
        temporary_block = Block()
        scaling_block = None
        if index > 0:
            scaling_block = Upscale2D(
                factor=self.factor_from_resolutions(self.resolutions[index - 1], self.resolutions[index])
            )
        temporary_block.extend(main_block.finalizingLayers())
        return ProgressiveStage(adversary=main_block, scaling_block=scaling_block, temporary_block=temporary_block)