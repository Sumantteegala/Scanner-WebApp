import cv2
import numpy as np
import math

kernal = np.ones((5, 5), dtype=float)


def getContours(cannyImg, img):
    contours, h = cv2.findContours(cannyImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    maxArea = 0
    biggest = np.array([])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        arclength = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * arclength, True)
        # cv2.drawContours(img, approx, -1, (255, 0, 255), 2)
        # if area > 5000:
        if area > maxArea and len(approx) == 4:
            biggest = approx
            maxArea = area
    # cv2.drawContours(img, biggest, -1, (255, 0, 255), 5)
    print(biggest)
    return biggest

def getWrap(imageCopy, approx):
    zero = np.zeros((4, 2), dtype="float32")
    newApprox = np.reshape(approx, (4, 2))
    newApprox = np.float32(newApprox)
    print(newApprox)

    sum = newApprox.sum(axis=1)
    zero[0] = approx[np.argmin(sum)]
    zero[3] = approx[np.argmax(sum)]
    diff = np.diff(newApprox, axis=1)
    zero[1] = approx[np.argmin(diff)]
    zero[2] = approx[np.argmax(diff)]

    (tl, tr, br, bl) = zero
    print(tl, tr, br, bl)
    widthBottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthTop = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightRight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightLeft = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxHeight = max(int(heightRight), int(heightLeft))
    maxWidth = max(int(widthBottom), int(widthTop))
    # print(width, height)
    newPoints = np.array([[0, 0],[0, maxHeight],[maxWidth, maxHeight], [maxWidth, 0]], dtype = "float32")
    print(newApprox)
    # computing the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(newApprox, newPoints)
    # print(m)
    newImg =  cv2.warpPerspective(imageCopy, m, (maxWidth, maxHeight))

    # return the wraped image
    return newImg

def input(img_url):
    print('---------------------', img_url)
    img = cv2.imread('static/uploads/test.jpeg')
    imageCopy = img.copy()
    img = cv2.resize(img, (600, 700))
    grayImg =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(img, (5, 5), 1)
    cannyImg = cv2.Canny(blurImg, 200, 200)
    imgDial = cv2.dilate(cannyImg, kernal, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernal, iterations=1)
    # imgdil = cv2.dilate(cannyImg, kernal, iterations=1)
    biggest = getContours(imgThreshold, img)
    finalImage = getWrap(imageCopy, biggest)
    # cv2.imshow("scanned image", finalImage)
    # cv2.imshow('original image', img)
    filename = 'static/uploads/final.jpeg'
    cv2.imwrite(filename, finalImage)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # print('--------------', filename[15:])
    return filename[15:]

# input('test.jpeg')
