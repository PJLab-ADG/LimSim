from simInfo.Evaluation import Decision_Evaluation
from simModel.Replay import ReplayModel
import argparse
parser = argparse.ArgumentParser(description='input dir')
parser.add_argument('--dir', type=str, help='replay dir', default='experiments/Scenarios/zadao/2024-01-24_17-26-09.db', required=False)

args = parser.parse_args()

database = args.dir
# 效果和用replay的效果不一样
model = ReplayModel(database)
evaluator = Decision_Evaluation(database, model.timeStep)
while not model.tpEnd:
    model.runStep()
    evaluator.Evaluate(model)