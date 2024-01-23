from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/zeroshot/gpt3.5/2024-01-23_19-41-26.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()