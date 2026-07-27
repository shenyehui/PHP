class JointCfg:
    def __init__(self, 
        name: str,
        rated_torque : float, 
        peak_torque  : float, 
        peak_velocity: float, 
        rated_power  : float,
        peak_power   : float = 0,
        armature     : float = 0, 
        friction     : float = 0,
        viscous_friction : float = 0,
    ):
        self.name = name
        self.rated_torque = rated_torque
        self.peak_torque = peak_torque
        self.peak_velocity = peak_velocity
        self.armature = armature
        self.rated_power = rated_power
        self.peak_power = peak_power
        self.friction = friction
        self.viscous_friction = viscous_friction
        
        if self.peak_power <= self.rated_power:
            # 假设峰值功率是3倍额定功率，且小于理想峰值功率（峰值力矩x峰值转速）3/4
            self.peak_power = min(rated_power*3.0, self.peak_torque*self.peak_velocity*0.75)
        
        vel_down = self.peak_power/self.peak_torque
        self.saturation_torque = self.peak_velocity * self.peak_torque/(self.peak_velocity - vel_down)

        print(
            f"[Joint Registered]: {name}, "
            f"rated torque: {rated_torque} Nm, "
            f"peak torque: {peak_torque} Nm, "
            f"peak velocity: {peak_velocity} rad/s, "
            f"rated power: {rated_power} W, "
            f"peak power: {self.peak_power} W, "
            f"saturation torque: {self.saturation_torque} Nm, "
            f"armature: {armature} kg*m^2, "
            f"friction: {friction}, "
            f"viscous friction: {viscous_friction}"
        )