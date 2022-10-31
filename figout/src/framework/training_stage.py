import tensorflow as tf


class TrainingStage:
    """
    TrainingStage is a class that abstracts the submodels of a multi-stage training session
    """
    def __init__(self):
        # Has the stage started
        self.__active = tf.Variable(False, trainable=False)
        # Is the stage the last one in the session
        self.__is_final = tf.Variable(False, trainable=False)
        # Is the stage the first one in the session
        self.__is_initial = tf.Variable(True, trainable=False)

    @property
    def active(self):
        return self.__active

    @active.setter
    def active(self, b):
        self.__active.assign(b)

    @property
    def is_final(self):
        return self.__is_final

    @is_final.setter
    def is_final(self, b):
        self.__is_final.assign(b)

    @property
    def is_initial(self):
        return self.__is_initial

    @is_initial.setter
    def is_initial(self, b):
        self.__is_initial.assign(b)


class TransitionTrainingStage(TrainingStage):
    """
    Some training stages undergo changes within the stage, called transitions
    """
    def __init__(self):
        super(TransitionTrainingStage, self).__init__()
        # Is the stage in transition
        self.__in_transition = tf.Variable(False, trainable=False)

    @property
    def in_transition(self):
        return self.__in_transition

    @in_transition.setter
    def in_transition(self, b):
        self.__in_transition.assign(b)


