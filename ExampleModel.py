from datetime import datetime

from simModel.Model import Model
from simModel.MPGUI import GUI

ego_id = '12'
sumo_gui = True
sumo_cfg_file = './networkFiles/CarlaTown05/Town05.sumocfg'
sumo_net_file = "./networkFiles/CarlaTown05/Town05.net.xml"
sumo_rou_file = "./networkFiles/CarlaTown05/carlavtypes.rou.xml,networkFiles/CarlaTown05/Town05.rou.xml"
carla_host = '127.0.0.1'
carla_port = 2000
step_length = 0.1
tls_manager = 'carla'
sync_vehicle_color = True
sync_vehicle_lights = True

stringTimestamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
database = './results/' + stringTimestamp + '.db'

if __name__ == '__main__':
    model = Model(
        egoID=ego_id, netFile=sumo_net_file, rouFile=sumo_rou_file,
        cfgFile=sumo_cfg_file, dataBase=database, SUMOGUI=sumo_gui,
        CARLACosim=True, carla_host=carla_host, carla_port=carla_port,
        tls_manager=tls_manager
    )
    model.start()

    gui = GUI(model)
    gui.start()

    while not model.tpEnd:
        model.moveStep()
        model.updateVeh()
    
    model.destroy()
    gui.terminate()
    gui.join()