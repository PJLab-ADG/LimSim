from rich import print

from simModel.egoTracking.replay import ReplayModel
from DriverAgent.Evaluation import GT_Evaluation

dataBase = '/home/PJLAB/leiwenjie/lwj/LimSimLLM/egoTrackingTest.db'

rmodel = ReplayModel(dataBase=dataBase,
                     startFrame=0)
evaluator = GT_Evaluation(dataBase, rmodel.timeStep)
while not rmodel.tpEnd:
    rmodel.moveStep()
    evaluator.Evaluate(rmodel)

rmodel.gui.destroy()
