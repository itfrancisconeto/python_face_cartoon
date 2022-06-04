import cv2
import numpy as np

class FaceEffect(object):
    
    def __init__(self, cap, faceCascade):
        self.cap = cap
        self.faceCascade = faceCascade

    """ Start of reference: https://www.analyticsvidhya.com/blog/2021/07/know-how-to-give-cartoon-effects-to-your-photos-with-opencv/ """

    def ColourQuantization(self, image, K=6)->any:
        '''Applied to color spaces; it is a process that reduces the number of distinct colors 
        used in an image, usually with the intention that the new image should be as visually 
        similar as possible to the original image'''
        Z = image.reshape((-1, 3)) 
        Z = np.float32(Z) 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        compactness, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        return res2

    def Countours(self, image)->any:
        '''Overlay the contours on the RGB image'''
        contoured_image = image
        gray = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2GRAY) 
        edged = cv2.Canny(gray, 200, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(contoured_image, contours, contourIdx=-1, color=6, thickness=1)
        return contoured_image   

    """ End of reference """

    def applyCartoonEffect(self)->None:
        while True:
            success, img = cap.read()
            faces = self.faceCascade.detectMultiScale(img, 1.2, 4)
            for (x, y, w, h) in faces:
                # get image
                imagem = img[y:y+h, x:x+w]
                # apply effect on image
                coloured = self.ColourQuantization(imagem)
                contoured = self.Countours(coloured)
                img[y:y+h, x:x+w] = contoured
            if len(faces) == 0:
                cv2.putText(img,'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            else:
                cv2.putText(img,'Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,205,50))
            cv2.imshow('Face Effect', img)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        self.cap.release()
        cv2.destroyWindow('Face Effect')     

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faceEffect = FaceEffect(cap, faceCascade)
    faceEffect.applyCartoonEffect()