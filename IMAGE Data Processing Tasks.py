#!/usr/bin/env python
# coding: utf-8

# # FILTERING AND BLURRING to extract Best FEATURES from Images

# In[ ]:


#Smoothing is used for NOISE FILTERING PURPOSE

#Filter are used for EDGE DETECTION . They use DIFFERENT VALUES in the kernel inorder to FETCH DIFFERENT EDGES .
#FOR DIAGONAL EDGE DETECTION , different valued kernal would be needed/used . For horizontal edge detection ,DIFFERENT valued Kernal would be needed . for VERTICAL EDGE DETECTION , VALUES in the Kernal would be Different 

#Filter are used for BLURING 
#Filter are used for NOISE REMOVING 
#Filter are used for ONLY BACKGROUD FOCUSED 
#Filter are SHARPENING particular Objects 

#1D SIGNALS and 1D IMAGES can be FILTERED using low pass filter  (USED for BLURING and REMOVING NOISES PURPOSES )
#1D SIGNALS and 1D IMAGES can be FILTERED using HIGH pass filter (USED for FINDING EDGES in the IMAGES )


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')


from matplotlib import pyplot as plt
import cv2 
import numpy as np


image = cv2.imread(r"C:\Users\90536\Desktop\images\lena.jpg" )

cvtImage = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

#Homogenous Filter : All pixels CONTRIBUTE equal weights . It is able to REDUCE NOISE and also did a little bluring of the image
#( Homogenous Filter : USED for BLURING and REMOVING NOISES PURPOSES )

KERNAL = np.ones((5,5) , np.float32)/25
dst = cv2.filter2D( cvtImage , -1 , KERNAL )

#AVERAGING Algorithm for Bluring 
BLUR = cv2.blur( cvtImage , (5 , 5 ))

#Gausian Filter : It uses Different WEIGHTED KERNAL  in x Direction and also uses Different WEIGHTED KERNAL  in y Direction . 
#For an KERNAL WINDOW , Pixels located at SIDE has LOWER WEIGHT wrt PIXELS located at the CENTER 
GAUSSIANBLUR = cv2.GaussianBlur(cvtImage , (5,5) , 0)

#Median Filter : It replaces Each Pixel Value by the MEDIAN OF THE NEIGHBOURING PIXELS .Good for SALT AND PEPPER NOISE (white dots and black dots ARE PRESENT on the IMAGE )
Median = cv2.medianBlur( cvtImage , 5)# 5 is a KERNAL_SIZE

#Bilatral Filter 
#To make EDGES SHARPER  . Edges became a little sharper wrt Other Filters  . It also REMOVES the Noise 
BilateralFilter = cv2.bilateralFilter(cvtImage , 9 , 75 ,75 )

titles =[ ' cvtImage' , 'HomogenousFilterImage' , 'blur' , 'GAUSSIANBLUR' ,'MedainFilter' , 'BilateralFilter']

images = [cvtImage ,dst  , BLUR , GAUSSIANBLUR , Median ,BilateralFilter]

for i in range(6) :
    plt.subplot( 2 , 3 , i+1 ) 
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

    
        
plt.show()


# # MORPHOLOGICAL TRANSFORMATIONS

# In[ ]:


#MORPHOLOGICAL TRANSFORMATION are OPERATIONS based on Image SHAPE . 

#MORPHOLOGICAL TRANSFORMATION are performed on BINARY IMAGES 

#Kernel is to CHANGE a PIXEL VALUE      by USING  NEARBY/NEIGHBOUR PIXELS



from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')


from matplotlib import pyplot as plt
import cv2 
import numpy as np


image = cv2.imread(r"C:\Users\90536\Desktop\images\smarties.png" , 0)

RET , MASK = cv2.threshold( image , 220 , 255 , cv2.THRESH_BINARY_INV)

kernal = np.ones( (2, 2) , np.uint8)
DILATION = cv2.dilate(MASK , kernal , iterations  = 1)    
# The shape of the OBJECTS is CHANGED . They became BIGGER and BIGGER if iteration value is INCREASED or kernal shape is INCREASED i.e (5 ,5 ) will PRODUCE bigger objects wrt (3,3) or (2,2)
EROSION  = cv2.erode( MASK , kernal ,iterations = 5 )   
# The shape of the OBJECTS DONT CHANGE .ORIGINAL SHAPE of the Object is retained . 
# Kernal Goes through EACH and EVERY PIXELS of an IMAGE . 
#Kernal gives 1 only if ALL the PIXELS in that particular KERNAL WINDOW are ALL 1 . Else it will GEVERATE 0. Thus black dots in an OBJECT BECAME BIGGER . 

OpenMorphologyEx = cv2.morphologyEx( MASK , cv2.MORPH_OPEN , kernal ,iterations =1 )
#It is erosion followed by dillation . It peforms quite well 

ClosingMorphologyEx = cv2.morphologyEx(MASK , cv2.MORPH_CLOSE , kernal , iterations =1)
#It is dillation followed by errosion .

GradientMorphologyEx = cv2.morphologyEx(MASK , cv2.MORPH_GRADIENT , kernal , iterations =1)
#Difference b/w DILate and Erode   THUS WE GET EDGE of an OBJECT 

TopHatMorphologyEx = cv2.morphologyEx(MASK , cv2.MORPH_TOPHAT , kernal , iterations =1)
#Difference b/w Mask and Opening of an mask


titles = [ 'Image1' , 'MASK' , 'diluteeeeee' , 'EROSIONNNNNN' , 'OpenMorphologyEx' , 'ClosingMorphologyEx' , 'GradientMorphologyEx' , 'TopHatMorphologyEx']
images = [ image  , MASK , DILATION , EROSION , OpenMorphologyEx ,ClosingMorphologyEx , GradientMorphologyEx ,  TopHatMorphologyEx]

for i in range(8) :
    plt.subplot( 3 , 3 , i+1 ) 
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    

    
    
plt.show()


# # VARYING RESOLUTION (Gaussian and Laplacian Pyramid )
# ## To reduce computational resources while processing images
# 

# In[ ]:


#We can create IMAGES of Different RESOLUTION . 
#PYRAMAID: Where an IMAGE is subject to    RRREPEATED SSSSMOOTHING and SUBSAMPLING . inorder to SCALE them to VARIOUS 
#By pyramid function , you can DOWN SCALE the Image . First it will REDUCE the RESOULATION and scale to HALF . AND SECOND STEP it will REDUCE the RESOULATION and SCALE to 1/4th of the ORIGIONAL IMAGE. THEN 1/8 of the ORIGIONAL IMAGE and then ... 1/16th of the ORIGIONAL IMAGE 

#GAUSAIN pyramid : RRREPEATED filtering and SUBSAMPLING ofan IMAGE . There are TWO functions pyrUp and pyrDown 

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'qt')


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\90536\Desktop\images\messi5.jpg" , 0 )

#Inorder to Copy an Image 
copyImage = image.copy()

pyrDownImage = [copyImage]
for i in range(6): #If we want to USE pyrDown 5 times OR we want to Reduce the RESOLUATION 5 times then use range(6)
    pyrDownImage1=cv2.pyrDown(copyImage)
    pyrDownImage.append(pyrDownImage1)
    copyImage = pyrDownImage1.copy()
    #cv2.imshow(str(i) , pyrDownImage1)
    
lastImage = pyrDownImage[5]


    
for i in range(5 , 0 ,-1 ):
    Gaussian_Extended=cv2.pyrUp(pyrDownImage[i])
    Laplacian=cv2.subtract(pyrDownImage[i-1] , Gaussian_Extended)
    cv2.imshow(str(i) ,Laplacian )
    
#cvtImage =cv2.cvtColor( image , cv2.COLOR_BGR2RGB)

pyrDownImageOnce = cv2.pyrDown(image)              # 1/4th of the origional Image 
pyrDownImageTwice = cv2.pyrDown(pyrDownImageOnce)   # 1/8th of the origional Image

#As you DECREASE the RESOLUTION of the Image by using pyrDown then INFORMATION(Color INFORMATION ) is LOST .
#Thus if we try to perform pyrUp on pyrDownImageTwice then IT WILL NOT BE SAME pyrDownImageOnce (Only will be same SIZE but will be BLURRED as it has lost the INFORMATION )

pyrUp = cv2.pyrUp(pyrDownImageTwice)

#cv2.imshow('pyrDownImageOnce' , pyrDownImageOnce)

#cv2.imshow('pyrDownImageTwice' , pyrDownImageTwice)

#cv2.imshow('pyrUp_Of_pyrDownImageTwice' , pyrUp)
cv2.imshow('image' , image)

if cv2.waitKey(0)== 27:
    cv2.destroyAllWindows()
    


# # Trackbar (For User to CHANGE VALUES at run time)
# 

# In[ ]:



import cv2
import numpy as np

def nothing(x): # x is a POSITION of the Trackbar
    print(x)

cv2.namedWindow('imageWindow')

image= np.zeros((300 , 512 ,3) , np.uint8)
cv2.createTrackbar('BLUE  ' ,  'imageWindow' , 0 , 255 ,nothing ) #Last value INDICATE THE FUNCTION which Executes if TRACKBAR Value CHANGES then 
cv2.createTrackbar('GREEN ' ,  'imageWindow' , 0 , 255 ,nothing )
cv2.createTrackbar('RED ' ,  'imageWindow' , 0 , 255 ,nothing )




while(1):
    cv2.imshow('imageWindow' , image)
    r = cv2.getTrackbarPos('RED','imageWindow')
    g = cv2.getTrackbarPos('GREEN','imageWindow')
    b = cv2.getTrackbarPos('BLUE','imageWindow')
    
    #b = cv2.getTrackbarPos('TrackbarName1BLUE' , 'image1Window')
    #g = cv2.getTrackbarPos('TrackbarName2GREEN' , 'image1Window') 
    #r = cv2.getTrackbarPos('TrackbarName3' , 'image1Window')
    image[:] = [b,g,r]
    
    k= cv2.waitKey(1)
    if k == 27:    
        break
    
    
    print(b )
    print(' , ')
    print(g)
    
cv2.destroyAllWindows()


# # HOUGH TRANSFORM to detect Lines and shape in Image 
# # Without builtin function of Hough Transform
# 

# In[ ]:


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    #Created hough space
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
# To find the votes for each angle and rho value in hough space
        for t_idx in range(num_thetas):
            
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rho 


# In[ ]:


def show_hough_line(img, accumulator, thetas, rhos ):
    import matplotlib.pyplot as plt

    fig, ax =plt.subplots(1, 2, figsize=(100, 100))
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(accumulator, cmap='jet')
    ax[1].set_aspect('auto', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles ')
    ax[1].set_ylabel('rho')
    ax[1].axis('image')
    lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength=20,maxLineGap=15)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("image" , image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    plt.show()
    
    
image = cv2.imread("C:\im02.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
edge_image = cv2.Canny(blur_image, 100, 200)
img = edge_image

accumulator, thetas, rhos = hough_line(img)
show_hough_line(img, accumulator,thetas, rhos )


# # DETECTING EDGES of an Image
# # without buitin Canny EDGE DETECTION Method

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


image = cv2.imread("C:\im03.png")
height = int(image.shape[0])
width = int(image.shape[1])

gray_image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)


Smoothed_image = cv2.GaussianBlur(gray_image , (5,5) , 1)


def edge_detection(image):
    
    kernelX = np.array([[-1 , 0 , 1 ] , [-2 , 0 , 2 ] , [-1 , 0 , 1]], np.float32 )

    kernalY = np.array([[ 1 , 2 , 1 ] , [ 0 , 0 , 0 ] , [-1 ,-2 ,-1]], np.float32 )
 
    Gx = convolve2d(Smoothed_image , kernelX)
    Gy = convolve2d(Smoothed_image , kernalY)
    G= np.hypot(Gx , Gy)


    #Normalized Magnitude
    Magnitude = G / G.max() 

    Angle = np.arctan2(Gx , Gy)
    
    return (Magnitude , Angle  )

Mag , Angle  = edge_detection(Smoothed_image)


Mag = np.array(Mag)


cv2.imshow("canny_edge_detection_image" , Mag)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Open and Capture Video from PC and apply thresholding to detect desirable features or Part of an Object in the Video

# In[ ]:


def nothing(x):
    pass

    
videoCapture=cv2.VideoCapture(0)    
cv2.namedWindow("Tracking")
cv2.createTrackbar('LH' , 'Tracking' , 0 , 255 , nothing)

cv2.createTrackbar("UH" , 'Tracking' , 255 , 255 , nothing)

cv2.createTrackbar("LS" , 'Tracking' , 0 , 255 , nothing)
cv2.createTrackbar("US" , 'Tracking' , 255 , 255 , nothing)
cv2.createTrackbar("LV" , 'Tracking' , 0 , 255 , nothing)
cv2.createTrackbar("UV" , 'Tracking' , 255 , 255 , nothing)



while(True):
    #image=cv2.imread(r'C:\Users\90536\Desktop\images\smarties.png')
    
    ret , image = videoCapture.read()
    
    LH=cv2.getTrackbarPos('LH' , 'Tracking')
    UH=cv2.getTrackbarPos('UH' , 'Tracking')
    LS=cv2.getTrackbarPos('LS' , 'Tracking')
    US=cv2.getTrackbarPos('US' , 'Tracking')
    LV=cv2.getTrackbarPos('LV' , 'Tracking')
    UV=cv2.getTrackbarPos('UV' , 'Tracking')
    
    conv_image = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
    
    print(LH)
    print(UH)
    print(LS)
    print(US)
    print(LV)
    print(UV)
    
    Lower_Thresold_ArrayValue_Blue_Color = np.array([LH , LS , LV])
    Upper_Thresold_ArrayValue_Blue_Color = np.array([UH , US , UV])
    
    mask= cv2.inRange (conv_image , Lower_Thresold_ArrayValue_Blue_Color , Upper_Thresold_ArrayValue_Blue_Color )    # Inorder to FETCH ONLY Blue values OF an Image .
    
    res = cv2.bitwise_and(image , image , mask = mask)
    
    cv2.imshow('frame' , image)
    cv2.imshow('mask' , mask)
    cv2.imshow('resultant', res)
    
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break

        
videoCapture.release()        
cv2.destroyAllWindows()
    
    
    
    
    


# # EVENTS

# In[ ]:


events = [ x for x in dir(cv2) if 'EVENT' in x]
print(events)


def newFunction ( event , x, y, flags , parameter):
    if event == cv2.EVENT_LBUTTONUP:
        blue = image[ y , x , 0]
        green = image[ y , x , 1]
        red = image[y , x , 2]
        print( x , '   ,   ' , y)
        text = str(blue) + ' , ' + str(green) + ' , ' + str(red)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image , text , (x,y) , font , 0.8 , (255 , 255 ,50) , 1 , cv2.LINE_AA)
        cv2.imshow('Window' , image)

image = cv2.imread('lena.jpg')  
cv2.imshow('Window' , image)

cv2.setMouseCallback( 'Window' , newFunction )

cv2.waitKey(0)
cv2.destroyAllWindows()


# # THRESHOLDING AND ADAPTIVE THRESHOLDING

# In[ ]:





# In[ ]:


#In THRESHOLDING technique , we have ONE THRESHOLD VALUE  which applies to ALL PIXELS in the image 
#In ADAPTIVE THRESHOLDING Technique , we have MANY THRESHOLD VALUE . for different REGIONS/SEGMENTS of an IMAGE 


#Inorder to OPEN THE MATPLOTLIB OUTPUT in the NEW WINDOW we will use these codes ELSE it's OUTPUT will not be OUTPUTED HERE.
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')

import cv2 
import numpy as np


#Thresholding is a Technique of SEPERATING an OBJECT from its BACKGROUND
#It divides the INPUT Pixels INTO 2 Groups . 1st Group has PIXELS having VALUE LESSSER than the threshold value and  
#2nd GROUP pIXELS will have VALUE GREATER than threshold value .


image = cv2.imread(r"C:\Users\90536\Desktop\images\sudoku.png" , 0)

ret , thresholded_image1 = cv2.threshold(image , 127 , 255 , cv2.THRESH_BINARY)# cv2.THRESH_BINARY is one of MANYYY Types of THRESHOLDING TECHNIQUES
ret , thresholded_image2 = cv2.threshold(image , 200 , 255 , cv2.THRESH_BINARY_INV)
#If the Color Pixel is Greater than THRESHOLD value(in this case is 200) then 
# In cv2.THRESH_BINARY OR cv2.THRESH_BINARY_INV , in BINARY only 2 SEGMENTS of PIXELS are made AFTER THRESHOLDING TECHNIQUE , 
#In cv2.THRESH_BINARY , The PIXELS having value GREATER than 200 will be assigned 100 pixel value (in THIS CASE ONLY as 100 value is assigned ) AND  0 value is assigned to PIXELS with pixel value LESSER than 200
#In cv2.THRESH_BINARY_ , The PIXELS having value LESSER than 200 will be assigned 100 pixel value (in THIS CASE ONLY as 100 value is assigned) AND  0 value is assigned to PIXELS with pixel value GREATER than 200


ret , thresholded_image3 = cv2.threshold(image , 175 , 255 , cv2.THRESH_TRUNC)
#if THRESOLDVALUE is 175 then LOWER than 175 pixels WILL REMAIN SAME(SAME as the ORIGINAL pixels ) AS BEFORE .There PIXEL value will not CHANGE at all
#and PIXELS having value GREATER than 175 pixel     WILL ALL converted to 175 pixel value. #the second value 255 has not much role HERE . 


ret , thresholded_image4 = cv2.threshold(image , 175 , 255 , cv2.THRESH_TOZERO)


#ADAPTIVE THRESOLDING : Different Thresolding Value for DIFFERENT REGION of the IMAGE 
#IT Calculates thresold value for SMALLER REGIONS and thus we have MULTIPLE THRESOLDVALUE for an IMAGE .
# THRESOLDING basically WORKS on ILLUMINATION LEVEL . Higher Brightness region are SHOWN whereas LOWER BRIGHTNESS region BECAME EITHER BLACK when applied Thresholding

ADTH1=cv2.adaptiveThreshold(image , 255 , cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY , 11 , 2  ) # cv.ADAPTIVE_THRESH_MEAN_C is the ( MEAN of the Neighbourhood BLOCKSIZE/AREA )
# It gives the MEAN of the  BLOCKSIZE * BLOCKSIZE  neighbourhood    MINUS C
#Adavptive METHOD will decide HOW THRESHOLDING value is calculated

#11 IS BLOCKSIZE
#2 IS C


ADTH2=cv2.adaptiveThreshold(image , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY , 11 ,2 )
#It gives the WEIGHTED SUM  of the  BLOCKSIZE * BLOCKSIZE  neighbourhood      MINUS C

cv2.imshow('THRESH_BINARYWindow1' ,thresholded_image1 )
cv2.imshow('THRESH_BINARY_INVWindow2' ,thresholded_image2 )
cv2.imshow('THRESH_TRUNCWindow3' , thresholded_image3 )

cv2.imshow('THRESH_TOZEROWindow4' , thresholded_image4 )

cv2.imshow('ADAPTIVE_THRESH_MEAN_CWindow5' , ADTH1)

cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_CWindow6' , ADTH2)


cv2.imshow('ImageWindow' , image)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


# # LAPLACE and SobelX  AND SobelY

# In[ ]:


#Image Gradient: DIRECTIONAL CHANGE in the (INTENSITY or THE COLOR ) in the Image 
#Laplacian Gradient was able to FIND EDGES of the Objects 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt')


image = cv2.imread(r"C:\Users\90536\Desktop\images\sudoku.png" , 0 )

#cvtImage =cv2.cvtColor( image , cv2.COLOR_BGR2RGB)

laplacian1 = cv2.Laplacian( image , cv2.CV_64F , ksize = 1) # We used cv2.CV_64F datatype as it Supports NEGATIVE VALUES TOO

#Now we will take Absolute Value of the LAPLACIAN TRANSFORMATION and CONVERT to UNSIGNED 8 bit integer 
laplacianAbsolute_KERNALSIZE1 = np.uint8(np.absolute(laplacian1))

laplacian2 = cv2.Laplacian( image , cv2.CV_64F , ksize = 5) # We used cv2.CV_64F datatype as it Supports NEGATIVE VALUES TOO

#Now we will take Absolute Value of the LAPLACIAN TRANSFORMATION and CONVERT to UNSIGNED 8 bit integer 
laplacianAbsolute_KERNALSIZE5 = np.uint8(np.absolute(laplacian2))

#Sobel 

sobelX =cv2.Sobel(image , cv2.CV_64F , 1  , 0 )# To tell the computer we want to use SobelX method we use 1 FOR dx 
#dx,direction accross x dimension .  Value is 1 IN THE ABOVE CASE
#dx,direction accross Y dimension .  Value is 0 IN THE ABOVE CASE

SobelY = cv2.Sobel(image , cv2.CV_64F , 0 , 1 )# To tell the computer we want to use SobelY method we use 1 FOR dy

#YOU CAN ALSO GIVE THE KERNALSIZE VALUE for Sobal as shown in Laplasian method  


#Now we will take Absolute Value of the LAPLACIAN TRANSFORMATION and CONVERT to UNSIGNED 8 bit integer 

sobelX = np.uint8(np.absolute(sobelX))

sobelY  = np.uint8(np.absolute(SobelY))

#YOU can also COMBINE the Results of sobelX and sobelY results to GET BOTH EDGES ALONG X and ALSO ALONG Y AXIS .
SOBEL_COMBINE_VIA_BITWISE_OR = cv2.bitwise_or(sobelX ,sobelY)

