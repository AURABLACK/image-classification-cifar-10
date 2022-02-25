from absl import flags

def define_flags():
    """
    Define flags for experiments
    """

    flags.DEFINE_string("train_dir", None, "")
    flags.define_string("data_dir", None, "")