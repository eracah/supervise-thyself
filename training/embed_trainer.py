from training.inference_trainer import InferenceTrainer


class EmbedTrainer(InferenceTrainer):
    def __init__(self, model, args, experiment):
        super(EmbedTrainer, self).__init__(model, args, experiment)
        
    def test(*args):
        assert False, "No test mode for Embedding"