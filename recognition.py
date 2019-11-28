#coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2

import networks # 网络结构
from decoding import Lexicon,wordBeamSearch,prefixBeamSearch,prefixMatch,bestPathDecode


print('read Chinese alphabet from file')
with open('data/chs_alphabet.txt','r') as f:
    alphabet = f.read().replace('\n', '')
alphabet = [alphabet, '0123456789X-.长期']

model = [networks.chsNet(1, len(alphabet[0])+1), networks.digitsNet(1, len(alphabet[1])+1)]
if torch.cuda.is_available():
    model[0] = model[0].cuda()
    model[1] = model[1].cuda()
print('loading pretrained model')
model[0].load_state_dict({k.replace('module.',''):v for k,v in torch.load('data/model_chs.pth').items()})
model[1].load_state_dict({k.replace('module.',''):v for k,v in torch.load('data/model_digits.pth').items()})

imgH = 22
def modelOutput(image, model):
    if image.shape[0] != imgH:
        image = cv2.resize(image,
                           (max(int(imgH*image.shape[1]/image.shape[0]),imgH), imgH),
                           cv2.INTER_LINEAR)
    image = torch.from_numpy(image.astype(np.float32))
    if torch.cuda.is_available():
        image = image.cuda()
    image = Variable(image.view(1, 1, *image.size()))

    model.eval()
    preds = model(image)
    preds = preds.view(preds.size(0), -1)
    preds = F.softmax(preds, dim=1)
    return preds.data

# 定义词典，存储地区码和地址的对应关系
lex_sex = ['男', '女']
lex_nation = ['仡佬','高山','藏','珞巴','景颇','门巴','仫佬','柯尔克孜',
          '畲','维吾尔','阿昌','瑶','裕固','撒拉','土','塔塔尔',
          '侗','傈僳','傣','崩龙','苗','达斡尔','羌','怒',
          '水','哈尼','乌孜别克','鄂温克','回','汉','赫哲','壮',
          '黎','布依','保安','土家','鄂伦春','佤','哈萨克','塔吉克',
          '毛难','俄罗斯','蒙古','纳西','独龙','东乡','布朗','拉祜',
          '普米','京','彝','朝鲜','满','白','基诺','锡伯']
lex_year = [str(i) for i in range(1958, 2009)]
lex_month = [str(i) for i in range(1, 13)]
lex_day = [str(i) for i in range(1, 32)]
lex_month_02d = ['%02d'%i for i in range(1, 13)]
lex_day_02d = ['%02d'%i for i in range(1, 32)]
lex_year_start = [str(i) for i in range(2009, 2020)]
lex_year_end = [str(i) for i in range(2014, 2040)]

lex_region = []
lex_code = []
region_code = {}
code_region = {}
for line in open('data/code_region.txt'):
    seg = line.strip().split(' ')
    if len(seg[0]) != 6:
        continue
    code_region[seg[0]] = seg[1]
    if seg[0][2:] == '0000':
        r1 = seg[1]
        lex_region.append(r1)
    elif seg[0][4:] == '00':
        r2 = seg[1]
        lex_region.append(r1+r2)
    else:
        if lex_code[-1][4:] == '00':
            lex_code.pop()
            lex_region.pop()
            if lex_code[-1][2:] == '0000':
                lex_code.pop()
                lex_region.pop()
        lex_region.append(r1+r2+seg[1])
    lex_code.append(seg[0])
for i in range(len(lex_code)):
    region_code[lex_region[i]] = lex_code[i]

def getRegionByCode(code):
    if code not in code_region:
        return None
    cs = code[:2] + '0000'
    rs = code_region[cs]
    if cs == code:
        return rs
    cs = code[:4] + '00'
    if len(code_region[cs]) > 1:
        rs += code_region[cs]
    if cs == code:
        return rs
    return rs + code_region[code]

def getBureauByCode(code):
    if code not in code_region:
        return None
    cs = code[:4] + '00'
    if len(code_region[cs]) > 1:
        rs = code_region[cs]
    else:
        rs = ''
    return rs + code_region[code] + '公安局'

lexicon = [Lexicon(lex_sex, alphabet[0]), Lexicon(lex_nation, alphabet[0]),
           Lexicon(lex_year, alphabet[1]), Lexicon(lex_month, alphabet[1]), Lexicon(lex_day, alphabet[1]),
           Lexicon(lex_month_02d, alphabet[1]), Lexicon(lex_day_02d, alphabet[1]),
           Lexicon(lex_region, alphabet[0]), Lexicon(lex_code, alphabet[1]),
           Lexicon(lex_year_start, alphabet[1]), Lexicon(lex_year_end, alphabet[1])]


def readTextImages(images, naive_decode = False):
    if naive_decode: # best-path decoding, without dictionary or rectifying
        name = bestPathDecode(modelOutput(images[0],model[0]),alphabet[0])
        sex = bestPathDecode(modelOutput(images[1],model[0]),alphabet[0])
        nation = bestPathDecode(modelOutput(images[2],model[0]),alphabet[0])
        year = bestPathDecode(modelOutput(images[3],model[1]),alphabet[1])
        month = bestPathDecode(modelOutput(images[4],model[1]),alphabet[1])
        day = bestPathDecode(modelOutput(images[5],model[1]),alphabet[1])
        address = bestPathDecode(modelOutput(images[6],model[0]),alphabet[0])
        id = bestPathDecode(modelOutput(images[7],model[1]),alphabet[1])+ \
             bestPathDecode(modelOutput(images[8], model[1]), alphabet[1])+ \
             bestPathDecode(modelOutput(images[9], model[1]), alphabet[1])+ \
             bestPathDecode(modelOutput(images[10], model[1]), alphabet[1])+ \
             bestPathDecode(modelOutput(images[11],model[1]),alphabet[1])
        psb = bestPathDecode(modelOutput(images[12],model[0]),alphabet[0])
        period = bestPathDecode(modelOutput(images[13], model[1]), alphabet[1]) + '.' + \
             bestPathDecode(modelOutput(images[14], model[1]), alphabet[1]) + '.' + \
             bestPathDecode(modelOutput(images[15], model[1]), alphabet[1]) + '-'
        if len(images) == 19:
            period += bestPathDecode(modelOutput(images[16], model[1]), alphabet[1]) + '.' + \
                 bestPathDecode(modelOutput(images[17], model[1]), alphabet[1]) + '.' + \
                 bestPathDecode(modelOutput(images[18], model[1]), alphabet[1])
        else:
            period += '长期'
        return [name, nation, sex, year, month, day, address, id, psb, period]

    name = bestPathDecode(modelOutput(images[0],model[0]),alphabet[0])
    sex,_ = wordBeamSearch(modelOutput(images[1],model[0]),lexicon[0])
    nation,_ = wordBeamSearch(modelOutput(images[2],model[0]),lexicon[1])
    id_tail = bestPathDecode(modelOutput(images[11],model[1]),alphabet[1])

    year,conf = wordBeamSearch(modelOutput(images[3],model[1]),lexicon[2])
    id_year,id_conf = wordBeamSearch(modelOutput(images[8],model[1]),lexicon[2])
    if year != id_year:
        if conf < id_conf:
            year = id_year
        else:
            id_year = year
    month,conf = wordBeamSearch(modelOutput(images[4],model[1]),lexicon[3])
    id_month,id_conf = wordBeamSearch(modelOutput(images[9],model[1]),lexicon[5])
    if int(month) != int(id_month):
        if conf < id_conf:
            month = str(int(id_month))
        else:
            id_month = '0' + month if len(month) == 1 else month
    day,conf = wordBeamSearch(modelOutput(images[5],model[1]),lexicon[4])
    id_day,id_conf = wordBeamSearch(modelOutput(images[10],model[1]),lexicon[6])
    if int(day) != int(id_day):
        if conf < id_conf:
            day = str(int(id_day))
        else:
            id_day = '0' + day if len(day) == 1 else day

    addr_output = modelOutput(images[6],model[0])
    region,conf,t = prefixBeamSearch(addr_output,lexicon[7])
    code_output = modelOutput(images[7],model[1])
    id_code,id_conf = wordBeamSearch(code_output,lexicon[8])
    if region not in region_code:
        region = getRegionByCode(id_code)
        t,_ = prefixMatch(addr_output, alphabet[0], region)
    elif region_code[region] != id_code:
        _,id_conf1 = prefixMatch(code_output, alphabet[1], region_code[region])
        t1,conf1 = prefixMatch(addr_output, alphabet[0], getRegionByCode(id_code))
        if conf*id_conf1 < conf1*id_conf: # code is more confident
            t = t1
            region = getRegionByCode(id_code)
        else: # region is more confident
            id_code = region_code[region]
    else: # just wanna align the region
        t,_ = prefixMatch(addr_output, alphabet[0], region)
    address = region + bestPathDecode(addr_output[t+1:],alphabet[0])
    id = id_code + id_year + id_month + id_day + id_tail
    psb = getBureauByCode(id_code)

    if len(images) == 16: # long term period
        period_year,_ = wordBeamSearch(modelOutput(images[13], model[1]), lexicon[9])
        period_month,_ = wordBeamSearch(modelOutput(images[14], model[1]), lexicon[5])
        period_day,_ = wordBeamSearch(modelOutput(images[15], model[1]), lexicon[6])
        period = period_year + '.' + period_month + '.' + period_day + '-长期'
    else:
        year_start_output = modelOutput(images[13],model[1])
        period_year,conf = wordBeamSearch(year_start_output,lexicon[9])
        year_end_output = modelOutput(images[16],model[1])
        period_year1,conf1 = wordBeamSearch(year_end_output,lexicon[10])
        if int(period_year1)-int(period_year) not in [5,10,20]:
            if conf < conf1:
                conf = 0
                for i in [5,10,20]:
                    candi_year = str(int(period_year1)-i)
                    _, candi_conf = prefixMatch(year_start_output, alphabet[1], candi_year)
                    if conf < candi_conf:
                        period_year = candi_year
                        conf = candi_conf
            else:
                conf1 = 0
                for i in [5,10,20]:
                    candi_year = str(int(period_year)+i)
                    _, candi_conf = prefixMatch(year_end_output, alphabet[1], candi_year)
                    if conf1 < candi_conf:
                        period_year1 = candi_year
                        conf1 = candi_conf

        period_month,conf = wordBeamSearch(modelOutput(images[14], model[1]), lexicon[5])
        period_month1,conf1 = wordBeamSearch(modelOutput(images[17], model[1]), lexicon[5])
        if period_month != period_month1 and conf < conf1:
            period_month = period_month1
        period_day,conf = wordBeamSearch(modelOutput(images[15], model[1]), lexicon[6])
        period_day1,conf1 = wordBeamSearch(modelOutput(images[18], model[1]), lexicon[6])
        if period_day != period_day1 and conf < conf1:
            period_day = period_day1

        period = '.' + period_month + '.' + period_day
        period = period_year + period + '-' + period_year1 + period

    return [name, nation, sex, year, month, day, address, id, psb, period]