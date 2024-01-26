from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/Scenarios/junction/2024-01-26_15-42-09.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()