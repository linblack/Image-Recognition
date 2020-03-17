import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

# cat_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\dogs-vs-cats\train'
# cat_img = cv2.imread(cat_dir+r'\cat.2.jpg')
# cat_img_encode = cv2.imencode('.jpg', cat_img)  #對影像進行JPEG編碼並暫存於記憶體，回傳tuple[True, []]
# cat_img_small = cv2.resize(cat_img, (300,300))  #改變尺寸(影像,(寬,高))
# cv2.imshow('Cat1', cat_img_small)
# k = cv2.waitKey(0)
# print(cat_img_small.shape)    #(396,312,3)
# cv2.imwrite('Cat1_s.jpg', cat_img_small)    #存檔(路徑, 影像)
# cv2.destroyAllWindows()

# capture = cv2.VideoCapture(0)  #建立攝影機0物件
# capture.isOpened()  #判斷攝影機是否開啟(True/False)
# success, img = capture.read()   #success(True/False), img影像
# capture.release()   #釋放攝影機


#無人車系統：道路辨識
# car_dir = r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\Car'
# car_img = cv2.imread(car_dir+r'\Car1.jpg')  #cv2.IMREAD_GRAYSCALE以灰階讀取
# gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)    #先進行灰階處理
# gauss = cv2.GaussianBlur(gray, (3,3), 0) #高斯模糊(影像, (高斯矩陣), 標準差)；矩陣越大越模糊
# canny = cv2.Canny(gauss, 50, 150)   #進行Canny邊緣偵測，(影像, 最低門檻, 最高門檻)；低於門檻輸出0[黑]不是邊緣，高於門檻輸出255[白]是邊緣，最低:最高=1:3
# mask = np.zeros_like(canny)         #全黑遮罩
# points = np.array([[[50,640],[690,640],[645,535],[430,535]]])   #(1,4,2)
# cv2.fillPoly(mask, points, 255)     #繪製多邊形(全黑遮罩, 感興趣座標點, 白色)
# roi = cv2.bitwise_and(canny, mask)  #像素AND運算，白+白=白
# lines = cv2.HoughLinesP(image=roi, rho=3, theta=np.pi/180, threshold=60, minLineLength=40, maxLineGap=50)
#                                 #Hough轉換，rho&theta為控制窗格大小，threshold穿過多少資料點，minLineLength線段最短長度(pixel)，maxLineGap點與點間距離
# lefts = []
# rights = []
# for line in lines:
#     line_p = line.reshape(4,)
#     x1, y1, x2, y2 = line_p
#     # cv2.line(car_img, (x1,y1), (x2,y2), (0,0,255), 3)
#     slope, b = np.polyfit((x1,x2),(y1,y2),1)    #取斜率及截距((X),(Y),幾次多項式)
#     if slope > 0:
#         lefts.append([slope, b])
#     else:
#         rights.append([slope, b])
#
# if rights and lefts:
#     right_avg = np.average(rights, axis=0)  #取右邊的平均直線，回傳[slope,b]
#     left_avg = np.average(lefts, axis=0)
#     avg_lines = np.array([right_avg, left_avg])
#
# sublines = []
# for line in avg_lines:
#     slope, b = line
#     y1 = car_img.shape[0]   #最底端
#     y2 = int(y1*(0.7))
#     x1 = int((y1-b)/slope)  #(y-b/m)取得線段X座標
#     x2 = int((y2-b)/slope)
#     sublines.append([x1,y1,x2,y2])
# sublines = np.array(sublines)   #(2,2)
#
# for line in sublines:
#     line_p = line.reshape(4,)   #(1,4)->(4,)
#     x1, y1, x2, y2 = line_p
#     cv2.line(car_img, (x1,y1), (x2,y2), (0,0,255), 3)   #(原圖，(起始點)，(結束點)，顏色(BGR)，線寬)
#
# cv2.imshow('Car1', car_img)     #顯示影像(視窗名稱,影像)
# # cv2.imshow('Car1_gray', gray)
# # cv2.imshow('Car1_gauss', gauss)
# # cv2.imshow('Car1_canny', canny)
# # cv2.imshow('Car1_mask', mask)
# # cv2.imshow('Car1_roi', roi)
# # # plt.imshow(canny)                   #使用pyplot.imshow畫圖來顯示座標
# # # plt.show()
# cv2.waitKey(0)                  #0代表無限,3000代表等待3000毫秒[3秒]
# cv2.destroyAllWindows()         #關閉所有視窗
#
#
# #移動偵測
# cap = cv2.VideoCapture(0)
# img_pre = None              #前影像預設為空
# while cap.isOpened():
#     success, img = cap.read()
#     if success:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_now = cv2.GaussianBlur(gray, (13,13), 5)
#         if img_pre is not None:
#             diff = cv2.absdiff(img_now, img_pre)    #前後影像相減
#             ret, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)   #門檻值:非黑即白，低於25為0[黑]，高於25為255[白]
#             _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #輪廓偵測，輸入(乾淨影像,mode,method)，輸出輪廓座標
#             if contours:
#                 cv2.drawContours(img, contours, -1, (255,255,255), 2)   #繪製輪廓，輸入(原始影像，輪廓座標，-1[全畫]，白色，寬度)
#         cv2.imshow('frame', img)
#         img_pre = img_now.copy()
#     k = cv2.waitKey(50)
#     if k == ord('q'):
#         cv2.destroyAllWindows()
#         cap.release()
#         break

#光學字元辨識(OCR, Optical Character Recognition)
img = cv2.imread(r'C:\Users\Blake\Desktop\OCR1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoise = cv2.fastNlMeansDenoising(gray, h=30)  #去除雜訊，(灰階影像，強度)
gaus = cv2.GaussianBlur(denoise, (5,5), 0)
ret, thresh = cv2.threshold(gaus, 30, 255, cv2.THRESH_BINARY)
cv2.imshow('THRESH', thresh)
cv2.waitKey(0)
text = pytesseract.image_to_string(thresh) #辨識圖片中的字元
print(text)