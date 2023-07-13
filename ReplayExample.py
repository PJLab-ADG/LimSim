from rich import print

from simModel.egoTracking.replay import ReplayModel


dataBase = 'egoTrackingTest.db'

rmodel = ReplayModel(dataBase=dataBase,
                     startFrame=0)

while not rmodel.tpEnd:
    rmodel.moveStep()

rmodel.gui.destroy()
