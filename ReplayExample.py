from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './experiments/zeroshot/GPT-4V/exp_1.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()