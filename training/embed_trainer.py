from training.inference_trainer import InferenceTrainer


class EmbeddingTrainer(InferenceTrainer):
    def __init__(self, model, args, experiment):
        super(EmbeddingTrainer, self).__init__(model, args, experiment)
        
    def test(*args):
        assert False, "No test mode for Embedding"