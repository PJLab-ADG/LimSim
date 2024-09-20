
class OpenDrive(object):

    def __init__(self):
        self._roads = {}
        self._junctions = {}



    @property
    def roads(self):
        return self._roads

    def getRoad(self, id):
        for road in self._roads:
            if road.id == id:
                return road

        return None


    @property
    def junctions(self):
        return self._junctions

