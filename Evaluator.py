from simInfo.Evaluation import Decision_Evaluation
from simModel.Replay import ReplayModel
if __name__ == "__main__":
    database = './results/2024-01-18_16-12-17.db'
    model = ReplayModel(database)
    evaluator = Decision_Evaluation(database, model.timeStep)
    while not model.tpEnd:
        model.runStep()
        evaluator.Evaluate(model)