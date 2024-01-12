from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results/2024-01-11_21-53-37.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()