def camera(
    sensor_id=0,
    width=640,
    height=480,
    displayW=960,
    displayH=540,
    fps=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sennsor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        %  (
            sensor_id,
            width,
            height,
            fps,
            flip_method,
            displayW,
            displayH,
        )
    )