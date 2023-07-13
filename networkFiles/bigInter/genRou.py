import random

routeEdges = [
    'E0 E1', 'E0 -E2', 'E0 E3',
    '-E1 -E2', '-E1 -E0', '-E1 E3',
    'E2 E1', 'E2 E3', 'E2 -E0',
    '-E3 -E0', '-E3 -E2', '-E3 E1'
]

routeNames = [
    'r_%i'%idx for idx in range(12)
]


def genRou(period: int):
    with open('bigInter.rou.xml', 'w', newline='') as f:
        print('''<routes>''', file=f)
        for r in range(12):
            print('''    <route id="{}" edges="{}"/>'''.format(
                routeNames[r], routeEdges[r]
            ),
            file=f)
        for i in range(int(3600/period)):
            print('''    <vehicle id="{}" depart="{}" route="{}"/>'''.format(
                str(i), float(i*period), random.choice(routeNames)
            ), 
            file=f)
        print('''</routes>''', file=f)

genRou(2)