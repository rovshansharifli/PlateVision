
class Results():
    def __init__(self,):
        self.results_dict = {}

    def update(self, id, text):
        # TODO: the dictionaries should be limited to avoid memory leakage
        self.results_dict[id] = text

    def get_plate(self, id):
        return self.results_dict[id] if id in self.results_dict.keys() else None


class TrackedVehicles():
    def __init__(self, ):
        self.tracked_vehicles = {}

    def get_non_rec_crops(self, ):
        non_rec_ids = []
        for tracked_veh_id in self.tracked_vehicles.keys():
            if self.tracked_vehicles[tracked_veh_id][0] == '':
                non_rec_ids.append({tracked_veh_id: self.tracked_vehicles[tracked_veh_id][1]})

        return non_rec_ids
    
    def get_crop_by_id(self, id):
        return self.tracked_vehicles[id][1]
    
    def add_tracked_vehicle(self, id, crop, frame_no):
        self.tracked_vehicles[id] = { 0: '',
                                      1: crop,
                                      2: frame_no}
        
    def remove_expired_tracked_vehicles(self, current_frame_no):
        expire_num_frame = 150
        expired_track_ids = []
        for tracked_veh_id in self.tracked_vehicles.keys():
            tracked_veh = self.tracked_vehicles[tracked_veh_id]
            if (tracked_veh[0] == '') and (current_frame_no - tracked_veh[2] > expire_num_frame):
                expired_track_ids.append(tracked_veh_id)

        for expired_id in expired_track_ids:
            del self.tracked_vehicles[expired_id]
    
    def set_plate_by_id(self, id, text):
        if id in self.tracked_vehicles.keys():
            self.tracked_vehicles[id][0] = text
        else:
            print('The id is not in the dict')

    def check_if_exists(self, id):
        if id in self.tracked_vehicles.keys():
            return True
        else:
            return False
    
    def get_results(self, ):
        return [t_veh[0] for t_veh in self.tracked_vehicles.values()]