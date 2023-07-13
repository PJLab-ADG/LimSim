from simModel.fixedScene.replay import ReplayModel

# dataBase = 'fixedSceneTest.db'
# dataBase = './fixedSceneTest_sc5.db'
dataBase = 'fixedSceneTest_copy.db'

frmodel = ReplayModel(dataBase)

while not frmodel.tpEnd:
    frmodel.moveStep()

frmodel.gui.destroy()
