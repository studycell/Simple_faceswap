import cv2
import os
import dlib
import numpy as np

current_path = os.getcwd()  # 获取当前路径
predictor_81_points_path = current_path + '/model/shape_predictor_81_face_landmarks.dat'

# 选取人脸81个特征点检测器
predictor_path = predictor_81_points_path
face_path = current_path + '/faces/'
detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1
predictor = dlib.shape_predictor(predictor_path)

def vtop(fromadd):
    capture = cv2.VideoCapture(fromadd)
    #capture = cv2.VideoCapture(0)
    c = 0
    if capture.isOpened():
        ret,frame = capture.read()
    else:
        ret = False
    while(ret):
        ret,frame = capture.read()
        if ret == False:
            break
        #cv2.imshow("results",frame)
        cv2.imwrite(current_path + "/picture_before/" + str(c) +  '.jpg',frame)
        c = c + 1
        cv2.waitKey(1)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    capture.release()

def ptov(size):
    fps = 24
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')

    #测试
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename = current_path + '/video_output.mp4',fourcc = fourcc,fps = fps,frameSize = (size[1],size[0]))

    for i in range(0,6000):
        p = i
        if os.path.exists(current_path + '/picture_after/' + str(p) + '.jpg'):
        #if os.path.exists(str(fromadd) + '/picture' + str(p) + '.jpg'):
            img = cv2.imread(current_path + '/picture_after/' + str(p) + '.jpg')
            #cv2.waitKey(100)
            #print(img.shape)
            video_writer.write(img)
            print(str(p) + '.jpg' + ' done')
        else:
            break
    video_writer.release()

class TooManyFaces(Exception):
    pass
class NoFace(Exception):
    pass

#face_rect = face_recognition.face_landmarks()

def get_landmark(image):
    face_rect = detector(image,1)
    #print(face_rect)
    #face_rect = face_recognition.face_landmarks(im)
    #print(face_rect)
    if len(face_rect) > 1:
        print('Too manyget_landmark faces.We only need no more than one face.')
        return -1
        #raise TooManyFaces
    elif len(face_rect) == 0:
        print('No face.We need at least one face.')
        return -1
        #raise NoFace
    else:
        #print(face_rect)
        #print(type(face_rect))
        #print(face_rect[0].left())
        #print(type(face_rect))
        #print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect[0].left(), face_rect[0].top(),
        #                                                       face_rect[0].right(), face_rect[0].bottom()))
        t = np.matrix([[p.x, p.y] for p in predictor(image, face_rect[0]).parts()])
        return t

#普氏分析
def transformation_from_points(points1,points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    #print(points1)
    #print(points2)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    #print(c1)
    #print(c2)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    #print(s1)
    #print(s2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])

#映射到对应的图像
def warp_im(im,M,dshape):
    output_im = np.zeros(dshape,dtype=im.dtype)
    cv2.warpAffine(im,M[:2],(dshape[1],dshape[0]),dst=output_im,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)
    return output_im


# 人脸特征点对应的器官
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 26))
RIGHT_BROW_POINTS = list(range(17,21))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 68))
left_points = [0,1,2,5,6,7,9,10,11,14,15,16]

top_points = list(range(69,81))

ALIGN_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,NOSE_POINTS + MOUTH_POINTS,left_points + top_points]

OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,NOSE_POINTS + MOUTH_POINTS,left_points + top_points]

FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6
SCALE_FACTOR = 1

#颜色矫正
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))
# 绘制凸包
def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color)

# 获取人脸掩模
def get_face_mask(img, landmarks):
    img = np.zeros(img.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(img, landmarks[group], color=1)
    img = np.array([img, img, img]).transpose((1, 2, 0))
    img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return img

#读入图片位置 返回图片和掩膜
def read_im_and_landmarks(fname,i):
    im1 = cv2.imread(fname, cv2.IMREAD_COLOR)
    #t = get_landmark(im1)

    if os.path.exists(current_path + '/picture_before/' + str(i + 1) + '.jpg'):
        im2 = cv2.imread(current_path + '/picture_before/' + str(i + 1) + '.jpg',cv2.IMREAD_COLOR)
        
        for j in range(0,im1.shape[0]):
            im1[j][0] = (im1[j][0] + im2[j][0]) / 2
            im1[j][1] = (im1[j][1] + im2[j][1]) / 2
            im1[j][2] = (im1[j][2] + im2[j][2]) / 2
            im2[j][0] = (im1[j][0] + im2[j][0]) / 2
            im2[j][1] = (im1[j][1] + im2[j][1]) / 2
            im2[j][2] = (im1[j][2] + im2[j][2]) / 2
            if(j > 300):
                break
        cv2.imwrite(current_path + '/picture_before/' + str(i + 1) + '.jpg',im2)
    else:
        print(1)
    t = get_landmark(im1)
    return im1,t

#对输出图像做N帧对均值
def doudong():
    i = 4
    while (os.path.exists(current_path + '/picture_after/' + str(i) + '.jpg')):
        im1 = cv2.imread(current_path + '/picture_after/' + str(i - 4) + '.jpg',cv2.IMREAD_COLOR)
        im2 = cv2.imread(current_path + '/picture_after/' + str(i - 3) + '.jpg',cv2.IMREAD_COLOR)
        im3 = cv2.imread(current_path + '/picture_after/' + str(i - 2) + '.jpg',cv2.IMREAD_COLOR)
        im4 = cv2.imread(current_path + '/picture_after/' + str(i - 1) + '.jpg',cv2.IMREAD_COLOR)
        im5 = cv2.imread(current_path + '/picture_after/' + str(i) + '.jpg',cv2.IMREAD_COLOR)
        #print(im1)
        for j in range(0,im1.shape[0]):
            im1[j][0] = (im1[j][0] + im2[j][0] + im3[j][0] + im4[j][0] + im5[j][0]) / 5
            im2[j][0] = (im1[j][0] + im2[j][0] + im3[j][0] + im4[j][0] + im5[j][0]) / 5
            im3[j][0] = (im1[j][0] + im2[j][0] + im3[j][0] + im4[j][0] + im5[j][0]) / 5
            im4[j][0] = (im1[j][0] + im2[j][0] + im3[j][0] + im4[j][0] + im5[j][0]) / 5
            im5[j][0] = (im1[j][0] + im2[j][0] + im3[j][0] + im4[j][0] + im5[j][0]) / 5
            im1[j][1] = (im1[j][1] + im2[j][1] + im3[j][1] + im4[j][1] + im5[j][1]) / 5
            im2[j][1] = (im1[j][1] + im2[j][1] + im3[j][1] + im4[j][1] + im5[j][1]) / 5
            im3[j][1] = (im1[j][1] + im2[j][1] + im3[j][1] + im4[j][1] + im5[j][1]) / 5
            im4[j][1] = (im1[j][1] + im2[j][1] + im3[j][1] + im4[j][1] + im5[j][1]) / 5
            im5[j][1] = (im1[j][1] + im2[j][1] + im3[j][1] + im4[j][1] + im5[j][1]) / 5
            im1[j][2] = (im1[j][2] + im2[j][2] + im3[j][2] + im4[j][2] + im5[j][2]) / 5
            im2[j][2] = (im1[j][2] + im2[j][2] + im3[j][2] + im4[j][2] + im5[j][2]) / 5
            im3[j][2] = (im1[j][2] + im2[j][2] + im3[j][2] + im4[j][2] + im5[j][2]) / 5
            im4[j][2] = (im1[j][2] + im2[j][2] + im3[j][2] + im4[j][2] + im5[j][2]) / 5
            im5[j][2] = (im1[j][2] + im2[j][2] + im3[j][2] + im4[j][2] + im5[j][2]) / 5
            im1[j][3] = (im1[j][3] + im2[j][3] + im3[j][3] + im4[j][3] + im5[j][3]) / 5
            im2[j][3] = (im1[j][3] + im2[j][3] + im3[j][3] + im4[j][3] + im5[j][3]) / 5
            im3[j][3] = (im1[j][3] + im2[j][3] + im3[j][3] + im4[j][3] + im5[j][3]) / 5
            im4[j][3] = (im1[j][3] + im2[j][3] + im3[j][3] + im4[j][3] + im5[j][3]) / 5
            im5[j][3] = (im1[j][3] + im2[j][3] + im3[j][3] + im4[j][3] + im5[j][3]) / 5
            if (j > 300):
                break
        cv2.imwrite(current_path + '/picture_after/' + str(i - 4) + '.jpg', im1)
        cv2.imwrite(current_path + '/picture_after/' + str(i - 3) + '.jpg', im2)
        cv2.imwrite(current_path + '/picture_after/' + str(i - 2) + '.jpg', im3)
        cv2.imwrite(current_path + '/picture_after/' + str(i - 1) + '.jpg', im4)
        cv2.imwrite(current_path + '/picture_after/' + str(i) + '.jpg', im5)
        i = i + 1


def testvideo():
    #imgtra,landmarks_tra = read_im_and_landmarks('/Users/caizhen/Desktop/faceswap/testpicture.jpg')
    imgtra = cv2.imread(current_path + '/testpicture.jpg',cv2.IMREAD_COLOR)
    imgtra = cv2.resize(imgtra, (imgtra.shape[1] * SCALE_FACTOR,imgtra.shape[0] * SCALE_FACTOR))
    t = cv2.GaussianBlur(imgtra,(11,11),0)
    t = cv2.Canny(imgtra,10,70)
    imgtra = cv2.resize(imgtra,(t.shape[1] * SCALE_FACTOR,t.shape[0] * SCALE_FACTOR))
    #cv2.imshow(imgtra)
    landmarks_tra = get_landmark(imgtra)
    #cv2.imshow("landmarks_tra",landmarks_tra)
    #print(landmarks_tra)
    #print(landmarks_tra)
    for i in range(0,6000):
        p = i
        if os.path.exists(current_path + '/picture_before/' + str(p) + '.jpg'):
        #if os.path.exists(str(fromadd) + '/picture' + str(p) + '.jpg'):
            imgadd = current_path + '/picture_before/' + str(p) + '.jpg'
            #print(imgadd)

            cv2.waitKey(100)
            im1,landmarks1 = read_im_and_landmarks(imgadd,p)

            #im1 = cv2.imread(imgadd, cv2.IMREAD_COLOR)
            #im1 = cv2.resize(im1, (im1.shape[1] * SCALE_FACTOR, im1.shape[0] * SCALE_FACTOR))
            #landmarks1 = face_recognition.face_landmarks(im1)
            if type(landmarks1) == int:
                cv2.imwrite(current_path + "/picture_after/" + str(i) + ".jpg", im1)
                continue
            #print(landmarks1)

            M = transformation_from_points(landmarks1,landmarks_tra)
            #print(M)
            mask = get_face_mask(imgtra,landmarks_tra)
            #mask = cv2.GaussianBlur(mask,(11,11),0)
            #mask = cv2.Canny(mask,10,70)
            #cv2.imshow("mask",mask)

            #cv2.imshow("mask",mask)
            #cv2.imshow("mask",mask)
            #cv2.imshow("mask",mask)
            warped_mask = warp_im(mask,M,im1.shape)
            #cv2.imshow("warped_mask",warped_mask)
            combined_mask = np.max([get_face_mask(im1,landmarks1),warped_mask],axis=0)
            #cv2.imwrite("combined_mask/" + str(i) + ".jpg",combined_mask)
            #cv2.imshow("combined_mask",combined_mask)
            warped_imtra = warp_im(imgtra,M,im1.shape)

            cv2.imwrite(current_path + "/warped_imtra/" + str(i) + ".jpg",warped_imtra)
            #cv2.imshow("warped_imtra",warped_imtra)
            #cv2.imshow('imshow',warped_mask)
            output_im = im1 * (1.0 - combined_mask) + warped_imtra * combined_mask

            #cv2.imshow("output_im",output_im)
            #cv2.imshow("output_im",output_im)
            for j in range(17):
                output_im[j,0] = (output_im[j,0] + im1[j,0]) / 2
                output_im[j,1] = (output_im[j,1] + im1[j,1]) / 2
                output_im[j,2] = (output_im[j,2] + im1[j,2]) / 2
            for j in range(69,82):
                output_im[j, 0] = (output_im[j, 0] + im1[j, 0]) / 2
                output_im[j, 1] = (output_im[j, 1] + im1[j, 1]) / 2
                output_im[j, 2] = (output_im[j, 2] + im1[j, 2]) / 2
            cv2.imwrite(current_path + "/picture_after/" + str(i) + ".jpg",output_im)
        else:
            break

#imgtest = cv2.imread("testpicture.jpg")
#IMAGE_SHAPE = imgtest.shape()


if __name__ == '__main__':

    vtop("testvideo.mov")
    #get_landmark_test()
    #filter_test()
    #testvideo()
    #doudong()
    img = cv2.imread(current_path + "/picture_after/0.jpg")
    #print(face_recognition.face_landmarks(img))
    size = img.shape[0:2]
    #t = cv2.GaussianBlur(img,(11,11),0)
    #t = cv2.Canny(img,10,70)
    #cv2.imshow("t",t)
    #cv2.waitKey(0)
    #imgtra = cv2.resize(img,(t.shape[1] * SCALE_FACTOR,t.shape[0] * SCALE_FACTOR))
    #cv2.imshow("imgtra",imgtra)
    #cv2.waitKey(0)
    testvideo()
    doudong()
    ptov(size)