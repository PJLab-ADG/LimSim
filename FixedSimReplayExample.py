from simModel.fixedScene.replay import ReplayModel

dataBase = 'fixedSceneTest.db'

frmodel = ReplayModel(dataBase)

while not frmodel.tpEnd:
    frmodel.moveStep()

frmodel.gui.destroy()
