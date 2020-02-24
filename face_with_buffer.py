import os
import cv2
import mtcnn
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


detector = MTCNN()

def highlight_faces(image_path, faces):
  # display image
    image = plt.imread(image_path)
    plt.imshow(image)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x-70, y-70), width+140, height+140,
                          fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()

def get_frames(path):
    cam = cv2.VideoCapture(path)
    path_split = os.path.split(path)
    name = path_split[1] 
    frame_folder = 'train_frames/' + name 
    print(frame_folder)
    if not os.path.exists(frame_folder): 
        os.makedirs(frame_folder)
        print('folder created')
    counter = 0
    trash = 0
    while(True): 
        ret, frame = cam.read()
        counter += 1
        if ret:
            frame_path = frame_folder +'/frame_' + str(counter)
            faces = detector.detect_faces(frame)
            #highlight_faces(frame_path, faces)
            f = 0
            for face in faces: 
                f += 1
                frame_save = frame_path + '_' + str(f) + '.jpg' 
                box = face['box']
                if f>1:
                    print(frame_path,' ',box)
                if(box[2]<100 or box[3]<100):
                    trash += 1
                    f -=1
                    break
                else:
                    ## Cropping face with some buffer
                    cropped_image = frame[max(box[1]-70,0):min(box[1]+box[3]+70, frame.shape[0]), 
                                          max(box[0]-70,0):min(box[0]+box[2]+70, frame.shape[1])]
                    ## resizing for hog
                    resized_image = cv2.resize(cropped_image, (128,64)) 
                    cv2.imwrite(frame_save, resized_image)
        else:
            break
            ##break
    cam.release() 
    cv2.destroyAllWindows()
    print(counter, " frames created & ", counter-trash, " frames saved" )

get_frames('dfdc_train_part_45/agzpasxmwv.mp4')