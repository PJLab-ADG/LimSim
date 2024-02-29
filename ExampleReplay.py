from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './results//2024-02-29_16-58-09.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()