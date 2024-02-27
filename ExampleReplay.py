from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-02-21_18-46-19.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()