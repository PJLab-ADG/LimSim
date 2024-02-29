from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-02-29_16-20-05.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()