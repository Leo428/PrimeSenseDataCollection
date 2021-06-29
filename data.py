from topic import ObjectListener
import csv
from datetime import datetime, time
import os
import time

class Collector():
    def __init__(self) -> None:
        self.max_frames = 100000
        self.object_listener = ObjectListener()
        from prime import PrimeSense
        import cv2
        self.cv2 = cv2
        self.primesense = PrimeSense()
        self.dir = './PrimeSenseData/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.mkdir(self.dir)

    # save frame to a file and return its path
    def save_img(self, count, frame):
        path = str(count) + '.jpg'
        self.cv2.imwrite(os.path.join(self.dir, path), frame)
        return path

    def get_object_xyz(self):
        return self.object_listener.get_state().qpos

    def save_to_csv(self, writer, path, xyz):
            writer.writerow([path, xyz])

    def run(self):
        done = False
        count = 0
        self.primesense.start_stream()
        time.sleep(3)
        with open(os.path.join(self.dir, 'data.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            while not done and count < self.max_frames:
                key = self.cv2.waitKey(1) & 255
                # get frame
                frame = self.primesense.get_rgb()
                self.cv2.imshow('Data Collector', frame)
                # get image path
                path = self.save_img(count, frame)
                
                # get xyz of object
                xyz = self.get_object_xyz()
                
                # save to csv file
                self.save_to_csv(writer, path, xyz)
                
                count += 1
                # terminate code
                if key == 27: 
                    print("\tESC key detected!")
                    done = True

                time.sleep(0.5)

        self.primesense.close()
        ## Release resources 
        self.cv2.destroyAllWindows()
        print("Terminated")

if __name__ == '__main__':
    try:
        c = Collector()
        c.run()
    except Exception as e:
        print(e)
        c.primesense.close()
