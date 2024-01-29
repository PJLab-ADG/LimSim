from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/zeroshot/GPT3.5-150/2024-01-29_13-56-20.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()