#coding=utf-8
import cv2
import numpy as np
from scipy import signal

# 预设身份证的宽高为445x280
card_size = (445, 280)
dst_pt = np.array([[0,0],[card_size[0]-1,0],\
                   [card_size[0]-1,card_size[1]-1],\
                   [0,card_size[1]-1]],np.float32)

# 正反面Laplace模板，用于边缘检测误差造成的平移矫正
# 用拉普拉斯而非二值图做模板，因为边缘信息更加丰富，定位更准确
tmp_front = np.load('data/front_laplace.npy')
tmp_back = np.load('data/back_laplace.npy')
# “仅限BDCI比赛使用”始终出现在左上角，用于旋转矫正
tmp_logo = np.load('data/left-top_logo.npy')
# 反面有效期限第二项，用于判断是否长期
tmp_validity = [np.load('data/validity_longterm.npy'),np.load('data/validity_date.npy')]

# 文本行候选框，rect: [top, bottom, left, right]
# 正面14个：姓名 + 性别 + 民族 + 年 + 月 + 日 + 地址x3 + 地区码 + 年 + 月 + 日 + 尾号
# 反面9个：公安局x2 + 年 + 月 + 日 + 第二项（‘长期’或‘年月日’） + 年 + 月 + 日
front_txtrgn = np.load('data/front_text_rect.npy')
back_txtrgn = np.load('data/back_text_rect.npy')

# 水印的Laplace模板用于精确定位
wm_filter = [np.load('data/watermark1_laplace.npy'), np.load('data/watermark2_laplace.npy')]
# 二值图模板用于生成蒙版以进行后续去水印操作
wm_img = [np.load('data/watermark1_solid.npy'), np.load('data/watermark2_solid.npy')]
wm_shape = [img.shape for img in wm_img]

kernel = np.array([[-0.75, -1, -0.75], [-1, 8, -1], [-0.75, -1, -0.75]], np.float32)  # 锐化
def eraseWatermark(img, laplace):
    # 判断水印是两者中的哪一个
    id = 0
    mask = cv2.filter2D(laplace/255, -1, kernel=wm_filter[0])
    mask1 = cv2.filter2D(laplace/255, -1, kernel=wm_filter[1])
    if np.max(mask1) > np.max(mask):
        mask = mask1
        id = 1
    # 卷积图最大值处即为水印位置
    irange, jrange = np.where(mask == np.max(mask))
    # 利用mask获取水印蒙版
    mask = np.zeros(mask.shape, np.uint8)
    for i in irange:
        for j in jrange:
            bi1 = i - (wm_shape[id][0] // 2)
            bi2 = bi1 + wm_shape[id][0]
            bj1 = j - (wm_shape[id][1] // 2)
            bj2 = bj1 + wm_shape[id][1]
            wi1 = 0
            wi2 = wm_shape[id][0]
            wj1 = 0
            wj2 = wm_shape[id][1]
            if bi1 < 0:
                wi1 = -bi1
                bi1 = 0
            elif bi2 > card_size[1]:
                wi2 -= bi2 - card_size[1]
                bi2 = card_size[1]
            if bj1 < 0:
                wj1 = -bj1
                bj1 = 0
            elif bj2 > card_size[0]:
                wj2 -= bj2 - card_size[0]
                bj2 = card_size[0]
            mask[bi1:bi2, bj1:bj2] += wm_img[id][wi1:wi2,wj1:wj2]

    # 假设水印的灰度值为mc，以不透明度alpha叠加
    # 原图的灰度值为b，叠加后的灰度值a可由下式计算
    # a = mc * alpha * transparency + b * (1 - alpha * transparency)
    # transparency是水印自带的透明通道信息，边缘处透明渐变
    transparency = mask / np.max(mask)
    # 获取水印部分和身份证全图上面积最大的灰度值
    mask[mask < 100] = 0
    hist_cv = cv2.calcHist([img], [0], mask, [256], [0, 256])
    a = np.where(hist_cv == np.max(hist_cv))[0][0]
    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])
    b = np.where(hist_cv == np.max(hist_cv))[0][0]
    mc = 50 # 经验猜测值
    alpha = (b-a)/(b-mc)
    # 若水印完全不透明，则无法进行反变换，故设置上限
    if alpha >= 0.95:
        alpha = 0.95
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if transparency[i,j]:
                trans = transparency[i,j]*alpha
                img[i,j] = max(min((img[i,j] - trans*mc) / (1-trans),b),20)
    # 图像去模糊，Laplace锐化
    img = cv2.filter2D(img, -1, kernel=kernel)
    # 为反变换后的灰度取值设置合理上限
    img[img > b] = b
    # img[img < 20] = 20
    return img

# 输入为整幅图及两个矩形（每个矩形包含四个顶点坐标）
# 返回值为[imgs]类型，包含19或16张文本行图像
# 正面12张：姓名 + 性别 + 民族 + 年+ 月 + 日 + 地区码 + 年 + 月 + 日 + 尾号
# 背面7或4张：公安局 + （年 + 月 + 日）x2 or x1
def getTextFromCard(img, cards):
    card_img = []
    card_binary = []
    card_laplace = []
    for i in range(2):
        cimg = cv2.warpPerspective(
            img, cv2.getPerspectiveTransform(cards[i], dst_pt), card_size)
        # Otsu's thresholding
        _, binary = cv2.threshold(
            cimg, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 使用二值图的左上区域进行旋转矫正
        tl_conv = signal.correlate2d(tmp_logo, binary[4:26,2:152], mode='valid')
        rb_conv = signal.convolve2d(tmp_logo, binary[254:276,293:443], mode='valid')
        if np.max(tl_conv) < np.max(rb_conv):
            cv2.flip(binary, -1, binary)
            cv2.flip(cimg, -1, cimg)
        # 图像去水印（包含锐化过程）
        laplace = cv2.Laplacian(cimg, -1, ksize=5)
        cimg = eraseWatermark(cimg, laplace)
        # 使用去水印之后的图像重新生成二值图
        _, binary = cv2.threshold(
            cimg, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 去水印锐化后的身份证正反面：用于截取文本行
        # 去水印后的二值图：用于缩紧文本行候选框的右边界
        # Laplace图像：用于正反面判断及平移矫正
        card_img.append(cimg)
        card_binary.append(binary)
        card_laplace.append(laplace)

    # 0为正面，1为背面
    front_conv = signal.correlate2d(tmp_front, card_laplace[0], mode='valid')
    back_conv = signal.correlate2d(tmp_back, card_laplace[1], mode='valid')
    score = np.max(front_conv) + np.max(back_conv)
    # 1为正面，0为背面
    front_conv1 = signal.correlate2d(tmp_front, card_laplace[1], mode='valid')#*front_norm
    back_conv1 = signal.correlate2d(tmp_back, card_laplace[0], mode='valid')#*back_norm
    score1 = np.max(front_conv1) + np.max(back_conv1)
    # 交换01位置，保证0为正面，1为背面
    if score < score1:
        card_binary.reverse()
        card_img.reverse()
        front_conv = front_conv1
        back_conv = back_conv1

    text_imgs = []
    # 身份证正面
    i,j = np.where(front_conv == np.max(front_conv))
    i = i[0]
    j = j[0]
    text_rect = front_txtrgn.copy()
    k = 0
    while k < 14:
        rgn = text_rect[k]
        rgn[0] -= i
        rgn[1] -= i
        rgn[2] -= j
        rgn[3] -= j
        if k in [0,2,6,7,8]:
            # 不定长文本，缩紧文本框右边界
            # Name, Nationality, Address x3
            hist = card_binary[0][rgn[0]:rgn[1],rgn[2]:rgn[3]].sum(axis=0) / 255
            # 对于地址第2、3行，若第一个字为空，则该行为空
            if k in [7, 8] and len(np.where(hist[4:24] >= 3)[0]) < 5:
                k = 9
                continue
            right = len(hist) - 6
            while right >= 0:
                # 我们认为文本区域：每列至少3个像素；20列以内超过3个像素的至少有5列
                if hist[right] < 3:
                    right -= 1
                elif len(np.where(hist[max(0,right-20):right] >= 3)[0]) < 5:
                    right -= 20
                else:
                    break
            if right < 0:
                k += 1
                continue
            right = len(hist) - 6 - right
            rgn[3] -= right

        if k in [7,8]:
            # 地址的多行文本图像水平拼接
            text_imgs[-1] = cv2.hconcat((text_imgs[-1],card_img[0][rgn[0]:rgn[1],rgn[2]:rgn[3]]))
        else:
            text_imgs.append(card_img[0][rgn[0]:rgn[1],rgn[2]:rgn[3]])

        # 对于地址第1、2行，若该行文本未满，则认为不存在下一行地址
        if k in [6,7] and right > 20:
            k = 9
            continue

        k += 1

    # 身份证反面
    i,j = np.where(back_conv == np.max(back_conv))
    i = i[0]
    j = j[0]
    text_rect = back_txtrgn.copy()
    k = 0
    while k < 9:
        rgn = text_rect[k]
        rgn[0] -= i
        rgn[1] -= i
        rgn[2] -= j
        rgn[3] -= j
        if k in [0,1]:
            # Bureau
            hist = card_binary[1][rgn[0]:rgn[1]-6,rgn[2]:rgn[3]].sum(axis=0) / 255
            if k == 1 and len(np.where(hist[4:24] >= 3)[0]) < 5:
                k = 2
                continue
            right = len(hist) - 6
            while right >= 0:
                if hist[right] < 3:
                    right -= 1
                elif len(np.where(hist[max(0,right-20):right] >= 3)[0]) < 5:
                    right -= 20
                else:
                    break
            if right < 0:
                k += 1
                continue
            right = len(hist) - 6 - right
            rgn[3] -= right
        elif k == 5:
            # 若有效期限第二项是“长期”而非日期，无需继续截取年月日
            vimg = card_binary[1][rgn[0]:rgn[1],rgn[2]:rgn[3]]
            if signal.correlate2d(tmp_validity[0], vimg, mode='valid')[0][0] >\
                    signal.correlate2d(tmp_validity[1], vimg, mode='valid')[0][0]:
                break
            else:
                k = 6
                continue

        if k == 1:
            text_imgs[-1] = cv2.hconcat((text_imgs[-1],card_img[1][rgn[0]:rgn[1],rgn[2]:rgn[3]]))
        else:
            text_imgs.append(card_img[1][rgn[0]:rgn[1],rgn[2]:rgn[3]])

        if k == 0 and right > 20:
            k = 2
            continue

        k += 1

    # for i in range(len(text_imgs)):
    #     cv2.imshow('text%d'%i,text_imgs[i])
    # cv2.waitKey(0)

    return text_imgs
