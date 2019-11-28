#coding=utf-8
import cv2
import numpy as np

# 两个角度之间的夹角
angle_thresh = 0.05
def ang(theta1, theta2):
    delta = abs(theta1-theta2) % np.pi
    if delta > np.pi/2.:
        delta = np.pi - delta
    return delta

# 计算两点间距的平方，省去开根号运算
dist_thresh = 250*250
def distance(pt1, pt2):
    dx = abs(pt1[0] - pt2[0])
    dy = abs(pt1[1] - pt2[1])
    return dx*dx + dy*dy

# 计算直线交点
def crovPt(line1, line2):
    r1 = line1[0]
    t1 = line1[1]
    r2 = line2[0]
    t2 = line2[1]
    sin1 = np.sin(t1)
    sin2 = np.sin(t2)
    cos1 = np.cos(t1)
    cos2 = np.cos(t2)
    div = sin1*cos2-sin2*cos1
    x = (r2*sin1-r1*sin2)/div
    y = (r1*cos2-r2*cos1)/div
    return (int(x),int(y))

def findParallelPairs(lines, dists):
    pair_ids = [[] for i in range(len(dists))]

    def insert(gid, pair):
        for p in pair_ids[gid]:
            if pair[0] == p[0]:
                if ang(lines[pair[1]][0], lines[p[0]][0]) < ang(lines[p[1]][0], lines[p[0]][0]):
                    p[1] = pair[1]
                return
            elif pair[1] == p[1]:
                if ang(lines[pair[0]][0], lines[p[1]][0]) < ang(lines[p[0]][0], lines[p[1]][0]):
                    p[0] = pair[0]
                return

        pair_ids[gid].append(pair)

    for i in range(1,len(lines)):
        # y*sin(theta) + x*cos(theta) = rho
        rho = lines[i][0] # 直线到原点的距离
        theta = lines[i][1] # 直线角度
        for j in range(i):
            r = lines[j][0]
            t = lines[j][1]
            if ang(t,theta) < angle_thresh:
                dist = abs(r-rho)
                for k in range(len(dists)):
                    if dists[k]-5 <= dist <= dists[k]+5:
                        if (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
                            if rho/np.cos(theta) < r/np.cos(t):
                                insert(k,[i,j])
                            else:
                                insert(k,[j,i])
                        else:
                            if rho/np.sin(theta) < r/np.sin(t):
                                insert(k,[i,j])
                            else:
                                insert(k,[j,i])

    lines = [[[lines[p[0]],lines[p[1]]] for p in group] for group in pair_ids]
    return lines[0] if len(lines) == 1 else lines

def findRectFromCoutour(edges, cnt):

    # 剔除底纹只余外轮廓，在此图上利用霍夫变换寻找直线
    hough_thresh = 100
    lines = cv2.HoughLines(edges,1,np.pi/180,hough_thresh)[:,0,:]
    # 根据平行线间距寻找身份证对边
    long_side, short_side = findParallelPairs(lines, [280, 445])
    # 若平行边数量不足，则降低霍夫阈值，得到更多候选直线
    while len(long_side) < cnt and len(short_side) < cnt:
        hough_thresh -= 20
        lines = cv2.HoughLines(edges,1,np.pi/180,hough_thresh)[:,0,:]
        long_side, short_side = findParallelPairs(lines, [280, 445])
    while len(long_side) < cnt:
        hough_thresh -= 20
        lines = cv2.HoughLines(edges,1,np.pi/180,hough_thresh)[:,0,:]
        long_side = findParallelPairs(lines, [280])
    while len(short_side) < cnt:
        hough_thresh -= 20
        lines = cv2.HoughLines(edges,1,np.pi/180,hough_thresh)[:,0,:]
        short_side = findParallelPairs(lines, [445])


    # 计算矩形四边与轮廓像素的重合度
    def validRatio(rect):
        valid = 0
        count = 0
        for i in range(4):
            pt1 = rect[i]
            pt2 = rect[(i+1)%4]
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if abs(dx) > abs(dy):
                count += abs(dx)
                xrange = range(pt1[0], pt2[0], 1 if dx < 0 else -1)
                k = dy / dx
                for x in xrange:
                    y = int(pt1[1] + k * (x - pt1[0]))
                    if edges[y, x] or edges[y - 1, x] or edges[y + 1, x] \
                            or edges[y - 2, x] or edges[y + 2, x]:
                        valid += 1
            else:
                count += abs(dy)
                yrange = range(pt1[1], pt2[1], 1 if dy < 0 else -1)
                k = dx / dy
                for y in yrange:
                    x = int(pt1[0] + k * (y - pt1[1]))
                    if edges[y, x] or edges[y, x - 1] or edges[y, x + 1] \
                            or edges[y, x - 2] or edges[y, x + 2]:
                        valid += 1
        return float(valid) / count


    rects = []
    for i in range(len(long_side)):
        a, b = long_side[i]
        for j in range(len(short_side)):
            c, d = short_side[j]

            # 内角与pi/2的差值，判断长短边是否垂直
            dt_ac = ang(a[1],c[1]+np.pi/2.)
            dt_ad = ang(a[1],d[1]+np.pi/2.)
            dt_bd = ang(b[1],d[1]+np.pi/2.)
            dt_bc = ang(b[1],c[1]+np.pi/2.)
            dt_mean = (dt_ac + dt_ad + dt_bd + dt_bc) / 4.
            # if dt_ac > angle_thresh or dt_ad > angle_thresh\
            #         or dt_bd > angle_thresh or dt_bc > angle_thresh:
            if dt_mean > angle_thresh:
                continue

            rect = [crovPt(a,c),crovPt(a,d),crovPt(b,d),crovPt(b,c)]
            center_pt = ((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) / 4,
                         (rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1]) / 4)
            # 是否与已有四边形重叠
            overlaped = False
            for k in range(len(rects)):
                if distance(center_pt, rects[k][1]) < dist_thresh:
                    # 重叠则保留与轮廓像素重合度较高的四边形
                    if rects[k][0] < 0:
                        rects[k][0] = validRatio(rects[k][2])
                    ratio = validRatio(rect)
                    if ratio > rects[k][0]:
                        rects[k] = [ratio, center_pt, rect]
                    overlaped = True
                    break
            if not overlaped:
                rects.append([-1, center_pt, rect])

    if len(rects) > cnt:
        for rect in rects:
            if rect[0] < 0:
                rect[0] = validRatio(rect[2])
        sort(rects, key=lambda x:x[0])
        rects = rects[-cnt:]

    # 对四边形的顶点进行微调，使其成为445x280的矩形
    rects = [np.array(rect[2], np.float32) for rect in rects]
    for rect in rects:
        for a,b in [(0,1),(2,3)]:
            dist = distance(rect[a],rect[b]) ** 0.5
            delta = (rect[a] - rect[b]) * (445 / dist - 1) / 2
            rect[a] += delta
            rect[b] -= delta
        for a,b in [(0,3),(1,2)]:
            dist = distance(rect[a],rect[b]) ** 0.5
            delta = (rect[a] - rect[b]) * (280 / dist - 1) / 2
            rect[a] += delta
            rect[b] -= delta

    return rects

def getCardVertexes(img):
    # 设置较低阈值，提取身份证底纹
    texture = cv2.Canny(img, 1, 10, apertureSize = 3)
    # 通过闭操作（膨胀-腐蚀）将底纹连接成完整区域
    kernel = np.ones((5,5),np.uint8)
    texture = cv2.morphologyEx(texture, cv2.MORPH_CLOSE, kernel)
    texture = 255 - texture
    # 获取连通区域轮廓
    _, contours, hierarchy = cv2.findContours(texture, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # 依据轮廓围成的面积筛选身份证区域
    isolates = []
    together = [] # 若正反面距离太近，则连通区域可能粘连
    for cont in contours:
        area = cv2.contourArea(cont)
        if 100000 < area < 130000:
            isolates.append(cont)
        elif 200000 < area < 260000:
            together.append(cont)

    # 绘制该区域轮廓
    if len(isolates) == 2:
        rects = []
        for cont in isolates:
            edges = np.zeros(img.shape,np.uint8)
            cv2.drawContours(edges, cont, -1, (255), 1)
            res = findRectFromCoutour(edges, 1)
            rects.extend(res)
    elif len(together) == 1:
        edges = np.zeros(img.shape,np.uint8)
        cv2.drawContours(edges, cont, -1, (255), 1)
        rects = findRectFromCoutour(edges, 2)
    else:
        return None

    # # 检测结果绘制
    # for rect in rects:
    #     cv2.line(img, tuple(rect[0]), tuple(rect[1]), (255))
    #     cv2.line(img, tuple(rect[1]), tuple(rect[2]), (200))
    #     cv2.line(img, tuple(rect[2]), tuple(rect[3]), (155))
    #     cv2.line(img, tuple(rect[3]), tuple(rect[0]), (100))
    # cv2.imshow('card', img)
    # cv2.waitKey(0)

    return rects
