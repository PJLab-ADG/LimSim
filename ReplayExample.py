from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/ablation/3-shot-11-mem/2024-01-27_23-41-20.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()