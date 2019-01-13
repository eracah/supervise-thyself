from training.inference_trainer import InferenceTrainer


class PredictionTrainer(InferenceTrainer):
    def __init__(self, model, args, experiment):
        super(PredictionTrainer, self).__init__(model, args, experiment)
        
    def test(self, test_set):
        self.one_epoch(test_set,mode="test")
        self.do_pca_corr(test_set, self.model)
        
    