from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-01-26_13-17-41.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()