import carla

class CarlaClient():
    """Proxy class that adds typing support for carla"""
    
    def __init__(self, host='localhost', port=2000, timeout=0.2) -> None:
        self.client = carla.Client(host, port)
        self.client.timeout(timeout)