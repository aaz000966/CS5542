
#This code was inspired direclty by the ICPs given in the class



#Importing cv2 ,matplotlib
import cv2
import matplotlib.pyplot as MyPyplt




# Inserting photos to the system,  sources: https://sakatavegetables.com/specialty-tomatoes/
#as well as converting them to CV2 RBG

MyImage1 = cv2.imread('1.jpg')

MyImage1 = cv2.cvtColor(MyImage1, cv2.COLOR_BGR2RGB)

MyImage2 = cv2.imread('2.jpg')

MyImage2 = cv2.cvtColor(MyImage2, cv2.COLOR_BGR2RGB)

#As descriped that this line is required for version-3
orb = cv2.ORB_create()

#detecting and computing keypoints on both images, des1,2 will be used later on.
ImgKP, des1 = orb.detectAndCompute(MyImage1, None)

ImgKP2, des2 = orb.detectAndCompute(MyImage2, None)

#new 2 images with the keypoints displayed
Img1AfrPrs = cv2.drawKeypoints(MyImage1,ImgKP,None) # Draw circles.

Img2AfrPrs = cv2.drawKeypoints(MyImage2,ImgKP2,None) # Draw circles.

MyPyplt.imshow(Img1AfrPrs)
MyPyplt.show()
MyPyplt.imshow(Img2AfrPrs)
MyPyplt.show()


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

 #creating a new image with matches on both and connected using drawMatches()
img_matches = cv2.drawMatches(MyImage1, ImgKP, MyImage2, ImgKP2, matches[:50], MyImage2, flags=2)

MyPyplt.imshow(img_matches);
MyPyplt.show()

