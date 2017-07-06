
from keras.models import Sequential,load_model
import cv2
import numpy as np
from scipy import ndimage
from vector import distance, pnt2line

import time
import sys
sys.path.append('code/')



output=[]
#ucitava podatke za prepoznavanje projeva
model = load_model('MNISTmodel.h5')
print "out "
#funckija za izdvanaje linije plave boje
def lineFinder(img):
    #min i max range plave boje
    BLUE_MIN = np.array([200, 0, 0], np.uint8)
    BLUE_MAX = np.array([255, 50, 50], np.uint8)

    mask = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
    # izdvaja sve boje sa slike preko maske
    imgl2 = 1.0 * mask

    # Funckija za HoughLinesP sa ovog sajta : http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    maxLineGap = 20
    minLineLength = 5
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    minArray=[];
    maxArray=[];

    minP = [1500, 1500];
    minIdx=0;

    maxP=[0,0];
    maxIdx=0;

    for line in lines:
        for points in line:
            minPoint=[points[0],points[1]]
            maxPoint=[points[2],points[3]]
            minArray.append(minPoint)
            maxArray.append(maxPoint)



    for minIndex in range(0,len(minArray)):
        if(minArray[minIndex][0]<minP[0]):
            minP=minArray[minIndex]
            minIdx =minIndex


    for maxIndex in range(0, len(maxArray)):
        if (maxArray[maxIndex][0] > maxP[0]):
            maxP = maxArray[maxIndex]
            maxIdx = maxIndex

    x1,y1 = minArray[minIdx]
    x2,y2 = maxArray[maxIdx]

    # provera da li dobro nadje liniju

    #slika = Image.open('video-0/frame0.jpg')
    #draw = ImageDraw.Draw(slika)
    #draw.point((x1, y1), 'red')
    #draw.point((x2, y2), 'red')
    #draw.line((x1, y1, x2, y2), fill=128)
    #slika.save('test.png')
    print "Line Coordinates: ",x1, y1, x2, y2
    return x1,y1,x2,y2



cc = -1
def getId():
    global cc
    cc += 1
    return cc

def printAll():
    global output
    for out in output:
        print "*************************"
        print
        print video + " Total Sum: " + str(sum)
        print
        print "*************************"


def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

sum=0;

def numberRecognition(img2):
    global sum

    WHITE_MIN = np.array([230, 230, 230])
    WHITE_MAX = np.array([255, 255, 255])

    mask = cv2.inRange(img2, WHITE_MIN, WHITE_MAX)
    img0 = 1.0 * mask
    img0 = cv2.dilate(img0, kernel)  # cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0, kernel)

    x, y = el['center']

    numberImage = img0[y - 14: y + 14, x - 14: x + 14]

    put = 'newImages/'+video+'/number-' + str(len(crossing)) + '.jpg'
    cv2.imwrite(put, numberImage)



    prediction = model.predict(numberImage.reshape(1, 784), verbose=1)
    value = np.argmax(prediction)
    print "Value of current number: " + str(value)
    sum += value
    print "Total Sum: " + str(sum)



def checkProcentage():
    res = []
    n = 0
    with open('Genericki projekat - level 2/res.txt') as file:
        data = file.read()
        lines = data.split('\n')
        for id, line in enumerate(lines):
            if (id > 0):
                cols = line.split('\t')
                if (cols[0] == ''):
                    continue
                cols[1] = cols[1].replace('\r', '')
                res.append(float(cols[1]))
                n += 1

    correct = 0
    student = []
    student_results = []
    with open("Genericki projekat - level 2/out.txt") as file:
        data = file.read()
        lines = data.split('\n')
        for id, line in enumerate(lines):
            cols = line.split('\t')
            if (cols[0] == ''):
                continue
            if (id == 0):
                student = cols
            elif (id > 1):
                cols[1] = cols[1].replace('\r', '')
                student_results.append(float(cols[1]))

    diff = 0
    for index, res_col in enumerate(res):
        diff += abs(res_col - student_results[index])
    percentage = 100 - diff / sum(res) * 100

    print student
    print 'Procenat tacnosti:\t' + str(percentage)
    print 'Ukupno:\t' + str(n)

outFile = open("Genericki projekat - level 2/out.txt", "w")
outFile.write("RA 121/2013 Radovan Bogojevic\n")
outFile.write("file\tsum\n")

for index in range(0,10):
    video="video-"+str(index)+".avi"
    cap = cv2.VideoCapture("Genericki projekat - level 2/"+video)
    print video
    kernel = np.ones((2,2),np.uint8)
    WHITE_MIN = np.array([230, 230, 230])
    WHITE_MAX = np.array([255, 255, 255])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('video-0-output.avi',fourcc, 20.0, (640,480))

    elements = []
    t =0
    times = []
    ret=True
    crossing=[]
    firstFrame=True

    while (ret):


        start_time = time.time()
        ret, img = cap.read()
        img2=img
        if (firstFrame):
            firstFrame=False
            x1, y1, x2, y2 = lineFinder(img)
            line = [(x1, y1), (x2, y2)]


        if (ret == True):
            WHITE_MIN = np.array(WHITE_MIN, dtype="uint8")
            WHITE_MAX = np.array(WHITE_MAX, dtype="uint8")
            mask = cv2.inRange(img, WHITE_MIN, WHITE_MAX)
            img0 = 1.0 * mask

            img0 = cv2.dilate(img0, kernel)
            img0 = cv2.dilate(img0, kernel)

            labeled, nr_objects = ndimage.label(img0)
            objects = ndimage.find_objects(labeled)
            for i in range(nr_objects):
                loc = objects[i]
                (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                            (loc[0].stop + loc[0].start) / 2)
                (dxc, dyc) = ((loc[1].stop - loc[1].start),
                              (loc[0].stop - loc[0].start))

                if (dxc > 11 or dyc > 11):

                    elem = {'center': (xc, yc), 'size': (dxc, dyc), 't': t}
                    # find in range
                    lst = inRange(20, elem, elements)
                    nn = len(lst)
                    if nn == 0:
                        elem['id'] = getId()
                        elem['t'] = t
                        elem['pass'] = False
                        elem['history'] = [{'center': (xc, yc), 'size': (dxc, dyc), 't': t}]

                        elements.append(elem)
                    elif nn == 1:
                        lst[0]['center'] = elem['center']
                        lst[0]['t'] = t
                        lst[0]['history'].append({'center': (xc, yc), 'size': (dxc, dyc), 't': t})


            for el in elements:
                tt = t - el['t']
                if (tt < 3):
                    dist, pnt, r = pnt2line(el['center'], line[0], line[1])
                    if r > 0:
                        cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                        c = (25, 25, 255)
                        if (dist < 9):
                            c = (0, 255, 160)
                            if el['pass'] == False:
                                el['pass'] = True
                                crossing.append(el)
                                numberRecognition(img2)


            elapsed_time = time.time() - start_time
            times.append(elapsed_time * 1000)

            cv2.putText(img, 'Suma: ' + str(sum), (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)

            t += 1

            cv2.imshow(video, img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            out.write(img)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    et = np.array(times)

    output.append([video,str(sum)])
    print "*************************"
    print
    print video+" Total Sum: "+str(sum)
    print
    print "*************************"
    outFile.write(video + "\t" + str(sum)+"\n")

    sum = 0
    elements = []
    t = 0
    times = []
    ret = True
    crossing = []
    firstFrame = True


outFile.close()

#printAll()

checkProcentage()