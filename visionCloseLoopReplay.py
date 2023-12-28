from rich import print

from simModel.egoTracking.VCLReplay import ReplayModel


dataBase = './results/2023-12-27_11-27-46.db'

rmodel = ReplayModel(dataBase=dataBase,
                     startFrame=0)

while not rmodel.tpEnd:
    rmodel.moveStep()

rmodel.gui.destroy()
