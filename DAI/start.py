import subprocess

def start_processes():
    detect_process = subprocess.Popen([
        'python', 'detect.py',
        '--weights', './runs/train/yolov7-tiny-continue-on-best-20-episodes/weights/best.pt',
        '--conf', '0.40',
        '--img-size', '416',
        '--source', '../CarlaData/images/test',
        '--snip-features',
        '--snip-folder-traffic-lights', '../classification/Detected_Traffic_Lights',
        '--snip-folder-traffic-signs', '../classification/Detected_Traffic_Signs'
    ])
    
    classify_process = subprocess.Popen(['python', '../classification/classify.py'])
    
    detect_process.wait()
    classify_process.wait()

if __name__ == "__main__":
    start_processes()
