from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/zeroshot/gpt4v/exp_9.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()