from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-01-10_14-47-30.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()