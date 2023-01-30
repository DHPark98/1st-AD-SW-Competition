# import process
import process_test
import sys
if __name__ == '__main__':
    play_name =sys.argv[1]
    if play_name == "Driving":
        processor = process_test.DoWork(play_name = "Driving", front_cam_name = "FRONT", rear_cam_name = "REAR", 
                                   rf_weight_file= "./model_weight_file/best_steering_model_0116.pth", 
                                   detect_weight_file="./model_weight_file/yolo_final_weight.pt")
        serial_result = processor.serial_start()
        if serial_result == True:
            front_camera_opened = processor.front_camera_start()
            if front_camera_opened == True:
                processor.Driving()

    elif play_name == "Parking":
        processor = process_test.DoWork(play_name = "Parking", front_cam_name = "FRONT", rear_cam_name = "REAR",
                                   detect_weight_file="./model_weight_file/yolo_final_weight.pt")
        serial_result = processor.serial_start()
        if serial_result == True:
            front_camera_opened = processor.front_camera_start()
            rear_camera_opened = processor.rear_camera_start()
            lidar_opened = processor.lidar_start()
            print("Lidar: ", lidar_opened)
            if front_camera_opened == True and rear_camera_opened == True:
                processor.Parking()