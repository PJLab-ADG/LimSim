from rich import print

from simModel.egoTracking.replay import ReplayModel


dataBase = 'egoTrackingTest.db'
# dataBase = 'ExpressB_success.db'
# dataBase = '2023-04-27_14-10-11_egoTracking_ir.db'

rmodel = ReplayModel(dataBase=dataBase,
                     startFrame=0)

while not rmodel.tpEnd:
    rmodel.moveStep()
    # if rmodel.canGetNextSce:
    #     print(rmodel.ego.availableLanes(rmodel.rb))

rmodel.gui.destroy()
