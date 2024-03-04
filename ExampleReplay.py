from simModel.Replay import ReplayModel
from simModel.RGUI import GUI

database = './ExampleDB/VLMAgentExample.db'
model = ReplayModel(database)
gui = GUI(model)

gui.run()