import time
import numpy as np
from dataclasses import dataclass
import cv2
import analytics

def camera_capture() -> np.array:
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)
    _, image = cam.read()
    return image

def kmeans_color_quantization(image, clusters=8, rounds=1):
    # get color means (kmeans) for shapes
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def apply_thresholding(image: np.array) -> tuple[np.array]:
    # Perform kmeans color segmentation into 2 clusters
    kmeans = kmeans_color_quantization(image, clusters=2)
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    # get thresholds on grayscale segmented image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return kmeans, gray, thresh

def find_contours(threshold_image: np.array):
    centers = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:][0]
    return centers

def draw_boxes(image: np.array, centers: list):
    # add box overlay to highlight specific particles
    for c in centers:
        rect = cv2.minAreaRect(c)
        box = np.intp(cv2.boxPoints(rect))
        #cv2.putText(image, str(cv2.contourArea(c)), rect, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.drawContours(image,[box],0,(0,0,255),2)
    return image

def gather_points(centers: list, threshold: int=2):
    #remove tiny specs using contour area filtering, gather points
    points_list = []
    size_list = []

    AREA_THRESHOLD = threshold
    for c in centers:
        area = cv2.contourArea(c)
        if area < AREA_THRESHOLD:
            cv2.drawContours(threshold, [c], -1, 0, -1)
        else:
            (x, y), radius = cv2.minEnclosingCircle(c)
            points_list.append((int(x), int(y)))
            size_list.append(area)
    
    return points_list, size_list

def image_overlay(image: np.array, threshold_image: np.array):
    # Overlay on original
    image[threshold_image==255] = (36,255,12)
    return image

@dataclass
class CoffeeAnalyzer:
    image: np.array
    thresh: np.array
    original: np.array
    points_list: np.array
    size_list: np.array

    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.original = self.image.copy()

    def _analyze(self):    
        kmeans, gray, thresh = apply_thresholding(self.image)
        centers = find_contours(threshold_image=thresh)
        rectangles = draw_boxes(self.image.copy(), centers)
        points_list, size_list = gather_points(centers)
        overlay = image_overlay(self.image.copy(), threshold_image=thresh)

        self.rectangles = rectangles
        self.overlay = overlay
        self.points_list = points_list
        self.size_list = size_list

    def _display(self):

        fontScale = 1
        thickness = 3
        
        # Using cv2.putText() method 
        self.overlay = cv2.putText(self.overlay, "Number of particles: {}".format(len(self.points_list)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , fontScale, (255, 0, 0) , thickness, cv2.LINE_AA)
        self.overlay = cv2.putText(self.overlay, "Average particle size: {:.3f}".format(sum(self.size_list)/len(self.size_list)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX , fontScale, (255, 0, 0) , thickness, cv2.LINE_AA)

        cv2.imshow('original', cv2.resize(self.original, (1600, 1600)))
        cv2.imshow('overlay', cv2.resize(self.overlay, (1600, 1600)))
        cv2.imshow('rects', cv2.resize(self.rectangles, (1600, 1600)))
        cv2.waitKey()

    def display_data(self):
        return analytics.particles_to_dataframe(self.points_list, self.size_list)


if __name__ == "__main__":

    ca = CoffeeAnalyzer("particles.jpg")
    ca._analyze()
    ca._display()
