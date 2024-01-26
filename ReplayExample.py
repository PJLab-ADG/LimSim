from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/zeroshot/gpt4v/2024-01-26_15-57-12.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()