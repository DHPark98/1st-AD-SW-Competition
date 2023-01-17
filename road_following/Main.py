import process
import sys
if __name__ == '__main__':
    
    processor = process.DoWork(play_name = "Driving", cam_name = "FRONT", rf_weight_file= "./model_weight_file/best_steering_model_0116.pth", detect_weight_file="./model_weight_file/yolo_weight.pt")
    serial_result = processor.serial_start()
    if serial_result == True:
        camera_opened = processor.camera_start()
        if camera_opened == True:
            processor.Dowork()

    
    