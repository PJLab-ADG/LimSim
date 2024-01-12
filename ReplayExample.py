from simModel.Replay import ReplayModel
from simModel.RGUI import GUI
from DriverAgent.Evaluation import Decision_Evaluation

database = './results/2024-01-11_21-53-37.db'
model = ReplayModel(database)
gui = GUI(model)
# evaluator = Decision_Evaluation(database, model.timeStep)
# while not model.tpEnd:
#     model.runStep()
#     evaluator.Evaluate(model)

gui.run()