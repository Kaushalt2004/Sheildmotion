# CARLA sensor setup utility for ADAS stack
import carla

def attach_sensors(world, vehicle):
    blueprint_library = world.get_blueprint_library()
    sensors = {}

    # RGB Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    sensors['camera'] = camera

    # LiDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '50')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    sensors['lidar'] = lidar

    # GNSS
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_transform = carla.Transform(carla.Location(x=0, z=2.8))
    gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
    sensors['gnss'] = gnss

    # IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_transform = carla.Transform(carla.Location(x=0, z=2.8))
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    sensors['imu'] = imu

    print("Sensors attached: camera, lidar, gnss, imu")
    return sensors
