import time

class AlertCooldownManager:
    def __init__(self, cooldowns):
        self.cooldowns = cooldowns
        self.last_times = {}

    def is_in_cooldown(self, cam_id, alert_type):
        key = f"{cam_id}_{alert_type}"
        now = time.time()
        last = self.last_times.get(key, 0)
        return now - last < self.cooldowns.get(alert_type, 300)

    def update(self, cam_id, alert_type):
        self.last_times[f"{cam_id}_{alert_type}"] = time.time()
