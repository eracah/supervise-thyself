from training.inference_trainer import InferenceTrainer


class PredictionTrainer(InferenceTrainer):
    def __init__(self, model, args, experiment):
        super(PredictionTrainer, self).__init__(model, args, experiment)
    