from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/zeroshot/gpt4v/2024-01-23_16-05-44.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()