# modules/estimator.py

class DistanceEstimator:
    def __init__(self, focal_length_y, camera_height_m, horizon_y):
        """
        :param focal_length_y: Tiêu cự trục dọc của camera (pixel) - f_y
        :param camera_height_m: Chiều cao lắp đặt camera so với mặt đất (mét) - C_h
        :param horizon_y: Tọa độ pixel trục dọc của đường chân trời - V_horizon - C_y
        """
        self.f_y = focal_length_y
        self.c_h = camera_height_m
        self.v_horizon = horizon_y

    def estimate(self, v_bottom_px):
        """
        Công thức: Z = (f_y * C_h) / (V_bottom - V_horizon)
        """
        if v_bottom_px <= self.v_horizon: 
            return -1.0 
        
        distance_m = (self.f_y * self.c_h) / (v_bottom_px - self.v_horizon)
        return distance_m