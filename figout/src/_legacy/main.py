import sys

from dependency_injector.wiring import inject, Provide
from _legacy.containers import MainContainer


@inject
def main(
        oneshot_callback=Provide[MainContainer.callback_container.oneshots_callback],
        graph_callback=Provide[MainContainer.callback_container.graph_callback],
        compiler=Provide[MainContainer.model_container.compiler],
        runner=Provide[MainContainer.training_container.runner],
):
    oneshot_callback.attach()
    # graph_callback.attach()
    compiler.compile()
    runner.run()


if __name__ == '__main__':
    mnist_gan = MainContainer()
    mnist_gan.config.from_yaml('resources/configs/mnist_aegan.yaml', required=True)
    mnist_gan.wire(modules=[sys.modules[__name__]])
    main()