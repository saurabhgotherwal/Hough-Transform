import numpy as np
import cv2


def TransformKernel(kernel):
    transform_kernel = kernel.copy()    
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            transform_kernel[i][j] = kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    return transform_kernel

def GetPadddedImage(image):       
    imagePadded = np.asarray([[ 0 for x in range(0,image.shape[1] + 2)] for y in range(0,image.shape[0] + 2)], dtype =np.uint8)
    imagePadded[1:(imagePadded.shape[0]-1), 1:(imagePadded.shape[1]-1)] = image 
    return imagePadded 

def Convolution(image, kernel):    
    kernel = TransformKernel(kernel)    
    imagePadded = GetPadddedImage(image)           
    imageConvolution = np.zeros(image.shape,dtype = np.float)    
    for i in range(1, imagePadded.shape[0]-1):
        for j in range(1, imagePadded.shape[1]-1):
            sum = 0            
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    sum += kernel[m][n]*imagePadded[i+m-1][j+n-1]
            imageConvolution[i-1][j-1] = np.absolute(sum)                           
    
    list = []
    for i in range(imageConvolution.shape[0]):
        for j in range(imageConvolution.shape[1]):
            if(image[i][j] < 0):
                imageConvolution[i][j] = imageConvolution[i][j] * -1
            list.append(imageConvolution[i][j])          
    imageConvolution /= max(list)
        
    return imageConvolution


def GetEdgeMagnitute(img1, img2):    
    img_copy = np.zeros(img1.shape)    
    
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q = (img1[i][j]**2 + img2[i][j]**2)**(1/2)
            if (q > 0.2):
                img_copy[i][j] = q
       
    
    return img_copy

def PerformSobel(image):
    kernelY = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]])
    gradientY = Convolution(image, kernelY)    
    kernelX = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    gradientX = Convolution(image, kernelX)        
    sobelEdgeMagnitute = GetEdgeMagnitute(gradientX,gradientY)
    return sobelEdgeMagnitute


def HoughLines(image,thetaResolution = 1, rhoResolution = 1):
    
    diagonalLenth = np.ceil(np.sqrt(image.shape[0] * image.shape[0] + image.shape[1] * image.shape[1]))   
    rhos = np.arange(-diagonalLenth, diagonalLenth,rhoResolution)    
    thetas = np.deg2rad(np.arange(-90.0, 90.0,thetaResolution))
    
    accumulator = np.zeros((int(2 * diagonalLenth), len(thetas)), dtype=np.uint64)
    y_indexes, x_indexes = np.nonzero(image)
    
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    
    for y,x in zip(y_indexes, x_indexes):        
        for thetaIndex in range(len(thetas)):            
            rho = round(x * cos[thetaIndex] + y * sin[thetaIndex]) + diagonalLenth
            accumulator[int(rho), thetaIndex] += 1

    return accumulator, thetas, rhos

def DrawAndFilterLines(image, rhos, thetas, accumulator, threshold, dividingFactor, minThetaRange, maxThetaRange):
    
    filteredRhoList = []
    filteredThetaList = []
    lineBucket = dict()      
    newList = []   
    
    locationIndex = np.where(accumulator > threshold)
        
    for rho, theta in zip(locationIndex[0],locationIndex[1]):
        filteredRhoList.append(rhos[rho]) 
        filteredThetaList.append(thetas[theta])   

    minimum = min(filteredRhoList)
    filteredRhoList -=  minimum    
    
    for i in range(len(filteredRhoList)):
        if (minThetaRange >filteredThetaList[i] > maxThetaRange):
            index = np.floor(filteredRhoList[i]/dividingFactor)            
            if index not in newList:
                lineBucket.setdefault(index, [])
                newList.append(index)
            lineBucket[index].append((filteredRhoList[i]+ minimum ,filteredThetaList[i])) 
    
    for bucketIndex in lineBucket:     
        bucket = lineBucket.get(bucketIndex)
        middle_line_index = int(len(bucket)/2)
        middleLine = bucket[middle_line_index]
        rho = middleLine[0]
        theta = middleLine[1]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 800*(-b))
        y1 = int(y0 + 800*(a))
        x2 = int(x0 - 800*(-b))
        y2 = int(y0 - 800*(a))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
            
    return image

def DetectLines(edges):
    accumulator, thetas, rhos = HoughLines(edges)   
        
    image = cv2.imread('hough.jpg',0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    redLines = DrawAndFilterLines(image, rhos, thetas, accumulator, threshold = 100, dividingFactor=100, minThetaRange = -0.03, maxThetaRange = -0.035)

    image = cv2.imread('hough.jpg',0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blueLines = DrawAndFilterLines(image, rhos, thetas, accumulator, threshold = 100, dividingFactor=70, minThetaRange = -0.62, maxThetaRange = -0.63)

    cv2.imwrite('red_line.jpg',redLines)
    cv2.imwrite('blue_line.jpg',blueLines)

    return

def HoughCircles(image, thetaResolution = 1): 
    
    rows = image.shape[0] 
    cols = image.shape[1]     
    thetas = np.deg2rad(np.arange(0, 360,thetaResolution))    
    sin = np.sin(thetas)
    cos = np.cos(thetas)     
    radius = [i for i in range(23,20,-1)] 
    radiusCount =  len(radius)   
    
    accumulator = np.zeros((rows,cols,radiusCount),dtype=np.uint64)
    y_indexes, x_indexes = np.nonzero(image)
    
    for r in range(radiusCount):          
        for row,column in zip(y_indexes, x_indexes):                           
            for thetaIndex in range(len(thetas)): 
                a = row - round(radius[r] * cos[thetaIndex]) 
                b = column - round(radius[r] * sin[thetaIndex])                
                if a >= 0 and a < rows and b >= 0 and b < cols: 
                    accumulator[int(a)][int(b)][r] += 1    
                    
    
   
    return accumulator, radius

def FilterCircles(accumulator, radius,threshold = 200):
    circles = []
    
    acc_cell_max = np.amax(accumulator)
    
    if(acc_cell_max > threshold): 
        # find the circles for this radius 
        for r in range(len(radius)):
            for i in range(accumulator.shape[0]): 
                for j in range(accumulator.shape[1]): 
                    if(accumulator[i][j][r] >= threshold):                        
                        circles.append((i,j,radius[r]))
                        accumulator[i-radius[r]:i+radius[r],j-radius[r]:j+radius[r]] = 0

        return circles

def DetectCircles(edges):
    circles = []
    radius = []
    accumulator = []

    accumulator, radius = HoughCircles(edges)

    circles = FilterCircles(accumulator, radius,threshold = 200)
    
    result = cv2.imread('hough.jpg',1)    
    for vertex in circles:
        cv2.circle(result,(vertex[1],vertex[0]),vertex[2],(0,0,255),2)   

    cv2.imwrite('coin.jpg',result)
    return


img = cv2.imread('hough.jpg',0)
edges = PerformSobel(img)

DetectLines(edges)
DetectCircles(edges)