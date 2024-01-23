from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-01-23_00-50-42.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()