from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-01-19_14-21-38.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()