from simInfo.Evaluation import Decision_Evaluation
from simModel.Replay import ReplayModel
if __name__ == "__main__":
    for i in range(10):
        database = f'./experiments/zeroshot/gpt4v/exp_{i}.db'
        model = ReplayModel(database)
        evaluator = Decision_Evaluation(database, model.timeStep)
        while not model.tpEnd:
            model.runStep()
            evaluator.Evaluate(model)