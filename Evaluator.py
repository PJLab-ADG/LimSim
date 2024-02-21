from simInfo.Evaluation import Decision_Evaluation
from simModel.Replay import ReplayModel

if __name__ == "__main__":
    database = f'./experiments/zeroshot/GPT-4V/exp_1.db'
    model = ReplayModel(database)
    evaluator = Decision_Evaluation(database, model.timeStep)
    while not model.tpEnd:
        model.runStep()
        evaluator.Evaluate(model)