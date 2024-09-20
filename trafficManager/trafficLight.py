class TrafficLight(object):
    def __init__(self, init_phase, time_green, time_yellow, time_red):
        self.init_phase = init_phase
        self.time_green = time_green
        self.time_yellow = time_yellow
        self.time_red = time_red
        self.currentState = ""

    def state_calculation(self, current_time):
        time_temp = (current_time + self.init_phase) % (
            self.time_yellow + self.time_green + self.time_red
        )
        if time_temp < self.time_green:
            self.currentState = "g"  # it means current state is green
        elif (
            time_temp > self.time_green
            and time_temp <= self.time_green + self.time_yellow
        ):
            self.currentState = "y"
        elif time_temp > self.time_green + self.time_yellow:
            self.currentState = "r"
