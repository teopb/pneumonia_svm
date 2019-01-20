#Some helper functions from https://www.kaggle.com/peterchang77/exploratory-data-analysis

import pandas as pd
import pydicom
import numpy as np
import pylab
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def parse_data(df, image_folder, lung_boxes):
    """
    Method to read a CSV file (Pandas dataframe) and parse the
    data into the following nested dictionary:

      parsed = {

        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'age': int,
            'sex': either 0 or 1 for male or female,
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # print("test1")
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid in lung_boxes:
            # print("test2")
            if pid not in parsed:
                # print("test3")
                dcm_file = '%s/%s.dcm' % (image_folder,pid)
                # dcm_data = pydicom.read_file(dcm_file)
                parsed[pid] = {
                    'dicom': dcm_file,
                    'label': row['Target'],
                    # 'sex' : dcm_data.PatientSex,
                    # 'age' : dcm_data.PatientAge,
                    'p_boxes': [],
                    'lung_boxes': lung_boxes[pid]['lungs']}

            # --- Add box if opacity is present
            if parsed[pid]['label'] == 1:
                parsed[pid]['p_boxes'].append(extract_box(row))
    # print(len(parsed))
    return parsed


def draw(data, predictions = None, sub = False):
    """
    Method to draw single patient with bounding box(es) if present

    """
    n = 8
    if sub is False:
        # --- Open DICOM file
        d = pydicom.read_file(data['dicom'])
        im = d.pixel_array
    else:
        im = data['im']

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    rgb1 = [220, 20, 60]
    rgb2 = [46, 139, 87]

    if sub is False:
        # --- Add pnuemonia boxes with random color if present
        for box in data['p_boxes']:
            # rgb = np.floor(np.random.rand(3) * 256).astype('int')
            im = overlay_box(im=im, box=box, rgb=rgb1, stroke=6)

        # --- Add lung boxes with random color if present
        for box in data['lung_boxes']:
            # rgb = np.floor(np.random.rand(3) * 256).astype('int')
            im = overlay_box(im=im, box=box, rgb=rgb2, stroke=6)

        if predictions is not None:
            for i in range(n*n):
                if predictions[i]:
                    overlay_color(im=im, index=i)
    # pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.imshow(im)
    pylab.axis('off')
    pylab.show()

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]

    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    # print('%s, %s, %s, %s' %(y1, y2, x1, x2))
    # print(rgb)

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

def overlay_color(im, index):
    """
    Method to overlay single box on image

    """

    # --- Extract coordinates
    n = 8
    l = 1024//n
    sx1 = int((index % n)*l)
    sx2 = int(sx1 + l)
    sy1 = int((index // n)*l)
    sy2 = int(sy1 + l)

    # rgb = cm.get_cmap(plt.get_cmap('Blues')).to_rgba([20])
    cmap = cm.get_cmap('winter')
    sm = cm.ScalarMappable(cmap=cmap)
    rgb = sm.to_rgba([1])
    # print("RGB:")
    # print(cmap)
    # print(rgb)

    for y in range(sy1, sy2):
        for x in range(sx1, sx2):
            grey = im[y, x][0]/255
            # print(grey)
            rgba = sm.to_rgba(grey, bytes=True, norm=False)
            # print(rgba)
            im[y, x][0] = rgba[0]
            im[y, x][1] = rgba[1]
            im[y, x][2] = rgba[2]

    return im
# def sub_image_bools(boxes):

def get_lung_boxes(lung_folder, file_list = None):
    size = 1024
    IDs = [];
    if file_list == None:
        for filename in os.listdir(lung_folder):
            if not filename.startswith('.'):
                ID, file_extension = os.path.splitext(filename)
                IDs.append(ID)
    else:
        for filename in file_list:
            IDs.append(filename)
    lung_boxes = {}
    for pid in IDs:
        if pid not in lung_boxes:
            lung_boxes[pid] = {'lungs':[]}
        f=open('%s/%s.txt' % (lung_folder, pid), 'r')
        f_lines = f.readlines()
        for line in f_lines:
            l = line.split()
            if len(l) == 5:
                h = int(float(l[4])*size)
                w = int(float(l[3])*size)
                y = int(float(l[2])*size) - h/2
                x = int(float(l[1])*size) - w/2
                lung_boxes[pid]['lungs'].append([y, x, h, w])
            elif len(l) == 4:
                # print(l)
                # print(pid)
                h = int(float(l[3])*size)
                w = int(float(l[2])*size)
                y = int(float(l[1])*size) - h/2
                x = int(float(l[0])*size) - w/2
                lung_boxes[pid]['lungs'].append([y, x, h, w])
        f.close()
    return lung_boxes

def in_box(index, boxes, partial):
    # partial is bool for whether partially in counts as less than fully in
    n = 8
    l = 1024//n
    sx1 = (index % n)*l
    sx2 = sx1 + l - 1
    sy1 = (index // n)*l
    sy2 = sy1 + l - 1

    for each in boxes:
        y1, x1, height, width = each
        y2 = y1 + height
        x2 = x1 + width
        # check each corner
        i1 = sx1 > x1 and sx1 < x2 and sy1 > y1 and sy1 < y2
        i2 = sx2 > x1 and sx2 < x2 and sy1 > y1 and sy1 < y2
        i3 = sx2 > x1 and sx2 < x2 and sy2 > y1 and sy2 < y2
        i4 = sx1 > x1 and sx1 < x2 and sy2 > y1 and sy2 < y2

        #all corners are in
        if (i1 and i2 and i3 and i4):
            return 1
        elif (i1 or i2 or i3 or i4):
            if partial:
                return 0.5
            else:
                return 1
    return 0

def split_to_sub_samples(parsed, full=True, uneven_split=True):
    n = 8
    size = 1024//n
    split_parsed = {}
    p_count=0
    np_count=0

    for each in iter(parsed):
        dcm_file = parsed[each]['dicom']
        dcm_data = pydicom.read_file(dcm_file)
        im = dcm_data.pixel_array
        h, w = im.shape
        new_im = (im.reshape(h//size, size, -1, size).swapaxes(1,2).reshape(-1, size, size))

        for i in range(n*n):
            pid = each + '_' + str(i)
            in_pnuemonia = in_box(i, parsed[each]['p_boxes'], False)
            in_lung = in_box(i, parsed[each]['lung_boxes'], True)

            if uneven_split or p_count > np_count or in_pnuemonia:
                if full or in_lung:
                    split_parsed[pid] = {
                        'pid': each,
                        'index': i,
                        'dicom': parsed[each]['dicom'],
                        'im': new_im[i],
                        'im_small': block_reduce(new_im[i], block_size=(size//4, size//4), func=np.mean).flatten(),
                        'label': parsed[each]['label'],
                        'p_boxes': in_pnuemonia,
                        'lung_boxes': in_lung,
                        'age': dcm_data.PatientAge,
                        'sex': 0 if dcm_data.PatientSex=='M' else 1}
                    if in_pnuemonia:
                        p_count += 1
                    else:
                        np_count += 1
    print("p_count: %s, np_count: %s" %(p_count, np_count))
    return split_parsed

def generate_samples(split_parsed, adj = False):
    # feature data in form img, age, sex, inlung
    # label data in form in_pnuemonia
    #TODO append and return IDs
    n = 8
    IDs = []
    samples = []
    labels = []
    for each in split_parsed:
        flat_img = split_parsed[each]['im'].flatten()
        age = split_parsed[each]['age']
        sex = split_parsed[each]['sex']
        in_lung = split_parsed[each]['lung_boxes']
        if adj:
            # find nearby values of i, i[0] = left and then clockwise ic = current
            adj_array = []
            pid = split_parsed[each]['pid']
            ic = split_parsed[each]['index']
            i = []
            i.append(ic-1)
            i.append(ic - n)
            i.append(ic + 1)
            i.append(ic + n)

            if (ic % n == 0):
                i[0] = -1

            elif ((ic + 1) % n == 0):
                i[2] = -1

            if(ic // n == 0):
                i[1] = -1

            elif(ic // n == n-1):
                i[3] = -1

            for j in range(4):
                if i[j] == -1:
                    adj_array.append(np.zeros(16))
                else:
                    adj_array.append(split_parsed[pid + '_' + str(i[j])]['im_small'])

            sample = np.append(flat_img, adj_array[0], adj_array[1], adj_array[2], adj_array[3] (age, sex, in_lung))
        else:
            sample = np.append(flat_img, (age, sex, in_lung))
        samples.append(sample)

        labels.append(split_parsed[each]['p_boxes'])

        IDs.append(each)

    #scale data before returning
    scaled_samples = preprocessing.scale(samples)
    return IDs, scaled_samples, labels
    # return IDs, np.array(scaled_samples), np.array(labels)

def score_on_positive(labels, predicted):
    total = 0
    correct = 0

    for label, prediction in zip(labels, predicted):
        if label == 1:
            total += 1
            if prediction == label:
                correct +=1

    return correct/total





df = pd.read_csv('all/stage_2_train_labels.csv')
#
# patientId = df['patientId'][0]
# dcm_file = 'all/stage_2_train_images/%s.dcm' % patientId
# dcm_data = pydicom.read_file(dcm_file)


# print(dcm_data)
#
# im = dcm_data.pixel_array
# # print(type(im))
# # print(im.dtype)
# # print(im.shape)
#
# pylab.imshow(im, cmap=pylab.cm.gist_gray)

# lung_boxes = get_lung_boxes('sarah_annotated_lungs')
# parsed = parse_data(df, 'all/stage_2_train_images', lung_boxes)

# pylab.axis('off')
#
# pylab.show()

# print(np.amax(im))

# parsed = parse_data(df, 'all/stage_2_train_images')
#
# dcm_data = pydicom.read_file(parsed['00436515-870c-4b36-a041-de91049b9ab4']['dicom'])
#
# print(dcm_data.PatientAge)
#
# print(df)
# ID = 'f0a1f49a-f67f-4716-97e0-840c83610f6f'
# draw(parsed[ID])

test_list = []
train_list = []
f=open('test_list.txt', 'r')
f_lines = f.readlines()
for each in f_lines:
    test_list.append(each.rstrip())
#
for filename in os.listdir('all/stage_2_train_images'):
    ID, file_extension = os.path.splitext(filename)
    if ID not in test_list and len(train_list) < 3001:
    # if ID not in test_list:
        train_list.append(ID)

# print(train_list)


#
print("Generating Test Set Method 1")
lung_boxes = get_lung_boxes('006_train_lungs', file_list = test_list)
# lung_boxes = get_lung_boxes('reduced1')
parsed = parse_data(df, 'all/stage_2_train_images', lung_boxes)
split_parsed1 = split_to_sub_samples(parsed)
testIDs, xtest2, ytest2 = generate_samples(split_parsed)
#
# print("Generating Test Set Method 2")
# lung_boxes = get_lung_boxes('boxresults_text_train', file_list = test_list)
# # lung_boxes = get_lung_boxes('reduced2')
# # print(len(lung_boxes))
# parsed = parse_data(df, 'all/stage_2_train_images', lung_boxes)
# split_parsed = split_to_sub_samples(parsed)
# testIDs, xtest2, ytest2 = generate_samples(split_parsed)
#
# print("Training with hand annotated")
# lung_boxes = get_lung_boxes('sarah_annotated_lungs')
# parsed = parse_data(df, 'all/stage_2_train_images', lung_boxes)
# split_parsed = split_to_sub_samples(parsed, full=True)
# predictIDs, x0, y0 = generate_samples(split_parsed)

# print(np.linalg.norm(x0[20]-x0[30]))

# C_2d_range = [1e-2, 1, 1e2]
# gamma_2d_range = [1e-1, 1, 1e1]
#
# print("Testing on Method 1 with linear svc")
# for c in C_2d_range:
#     for gamma in gamma_2d_range:
#         print("C = " + str(c) + " gamma = " + str(gamma))
#         model = svm.SVC(C = c, kernel = 'linear', gamma = gamma)
#         model.fit(x0, y0)
#         predicted = model.predict(xtest2)
#         # np.savez('output1', predictIDs, x0, y0, predicted)
#         total_score = accuracy_score(ytest2, predicted)
#         print("Score on total: " + str(total_score))
#
#         positive_score = score_on_positive(ytest2, predicted)
#         print("Score on positive: " + str(positive_score))

# data = np.load('output1.npz')

# print(np.average(data['arr_3']))
# print(np.average(data['arr_2']))

# sample_predictions = [
# 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 1, 1, 1, 0, 0, 0, 0,
# 0, 1, 1, 1, 0, 0, 0, 0,
# 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0, 0, 0, 0, 0]

# for each in parsed:
#     # ID = data['predictIDs'][i*64].split('_')[0]
#     # drawPredictions = data['predicted'][64*i:64*(1+1)]
#     print(each)
#     draw(parsed[each])
#     pylab.show()

# for each in parsed:
#     print(each)
#     print(parsed[each]['label'])
#     if parsed[each]['label']:
#         draw(parsed[each])
# test_id = 'f002b136-7acc-42e3-b989-071c6266fd60'



# draw(split_parsed['f002b136-7acc-42e3-b989-071c6266fd60_27'], sub=True)


# f002b136-7acc-42e3-b989-071c6266fd60


# ID= 'f002b136-7acc-42e3-b989-071c6266fd60'
#
# draw(parsed[ID], sample_predictions)
# pylab.show()



# print("Score: %s" % score)
# print("Testing on Method 2")
# predicted = model.predict(xtest2)
# score = accuracy_score(ytest2, predicted)
# print("Score: %s" % score)

# print("Generating Train Set Method 1")
# lung_boxes = get_lung_boxes('006_train_lungs', file_list = train_list)
# parsed = parse_data(df, 'all/stage_2_train_images', lung_boxes)
# split_parsed = split_to_sub_samples(parsed)
# xtrain1, ytrain1 = generate_samples(split_parsed)
# model = svm.SVC()
# model.fit(xtrain1, ytrain1)
# print("Testing on Method 1")
# predicted = model.predict(xtest1)
# score = accuracy_score(ytest1, predicted)
# print("Score: %s" % score)

print("Generating Train Set Method 1 even split")
# lung_boxes = get_lung_boxes('006_train_lungs', file_list = train_list)
# lung_boxes = get_lung_boxes('boxresults_text_train', file_list = train_list)
lung_boxes = get_lung_boxes('006_train_lungs', file_list = train_list)
parsed = parse_data(df, 'all/stage_2_train_images', lung_boxes)
split_parsed = split_to_sub_samples(parsed, full=False, uneven_split=False)
trainIDs, xtrain2, ytrain2 = generate_samples(split_parsed)
model = svm.SVC(C=1)
print("fiting on Method 1")
model.fit(xtrain2, ytrain2)
print("Testing on Method 1")
predicted = model.predict(xtest2)
score = accuracy_score(ytest2, predicted)
positive_score = score_on_positive(ytest2, predicted)
print("Score on positive: " + str(positive_score))
print("Score: %s" % score)
np.savez('output_full_for_print', xtrain2, ytrain2, testIDs, xtest2, ytest2, predicted)

for each in test_list:
    test_id = each
    pnuemonia_labels = []
    lung_labels = []
    for i in range(64):
        id = test_id +'_' +str(i)
        lung_labels.append(split_parsed[id]['lung_boxes'])
        pnuemonia_labels.append(split_parsed[id]['p_boxes'])
        if i % 8 == 0:
            print('\n')
        print(pnuemonia_labels[i], end = ' ')

    # draw(parsed[test_id], lung_labels)
    draw(parsed[test_id], pnuemonia_labels)
#
# draw(parsed['f0a1f49a-f67f-4716-97e0-840c83610f6f'])
# pylab.show()
# im = split_parsed['f0a1f49a-f67f-4716-97e0-840c83610f6f_2']['im']
#
# im = np.stack([im] * 3, axis=2)
#
# pylab.imshow(im, cmap=pylab.cm.gist_gray)
# pylab.axis('off')

# print(len(split_parsed))

# pylab.show()
