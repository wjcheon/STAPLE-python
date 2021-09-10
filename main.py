import os
import numpy as np
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
import SimpleITK as sitk
import copy
from calConsensus_standardalone import calConsensus_standardalone
import cv2
from tqdm import tqdm

mainPath = r"\\172.20.202.87\Users\keem\staple\models\model2";
targetWeight = 0.2
kidneyWeight =0.3 ## next is 0.65
savePath = r"\\172.20.202.87\Users\keem\staple\Final_210909_staple_result_model2_target_{0:1.2f}_kidney_{1:1.2f}".format(targetWeight, kidneyWeight)
try:
    os.stat(savePath)
except:
    os.mkdir(savePath)

folderList = os.listdir(mainPath);
patientNumber = 83;
reader = sitk.ImageFileReader()
reader.SetImageIO("PNGImageIO")
print("STAPLE process is starting...")
kidneySum =[]
tumorSum =[]
for iterPatient in tqdm(range(0, patientNumber)):

    dataPerPatient_kidney  = {}  # model 별 stack 쌓기
    dataPerPatient_tumor = {}  # model 별 stack 쌓기
    testFolderName = r"test{0:03d}".format(iterPatient+1)
    for modelTemp in folderList:
        #print(modelTemp)

        pathPerPatient = os.path.join(mainPath, modelTemp, testFolderName)
        sliceList = os.listdir(pathPerPatient)
        numSlices = np.shape(sliceList)[0]
        patientStack_kidney = []
        patientStack_tumor = []
        for iterSlice in range(0, numSlices):
            #print(iterSlice)
            imagePathTemp = os.path.join(pathPerPatient, sliceList[iterSlice])
            #imagePathTemp = '00023.png'
            imgTemp = sitk.ReadImage(imagePathTemp)
            imgTemp_np = sitk.GetArrayFromImage(imgTemp)

            imgTemp_np_kidney = copy.deepcopy(imgTemp_np)
            imgTemp_np_kidney = np.uint8(np.equal(imgTemp_np_kidney,1))

            imgTemp_np_tumor = copy.deepcopy(imgTemp_np)
            imgTemp_np_tumor = np.uint8(np.equal(imgTemp_np_tumor, 2))

            imgTemp_np_kidney_F = np.flipud(np.rot90(imgTemp_np_kidney))
            imgTemp_np_tumor_F = np.flipud(np.rot90(imgTemp_np_tumor))

            patientStack_kidney.append(imgTemp_np_kidney_F)
            patientStack_tumor.append(imgTemp_np_tumor_F)

        patientStack_kidney_swap = np.swapaxes(patientStack_kidney, axis1=0, axis2=2)
        imgTemp_np_tumor_swap = np.swapaxes(patientStack_tumor, axis1=0, axis2=2)
        dataPerPatient_kidney[modelTemp] = patientStack_kidney_swap
        dataPerPatient_tumor[modelTemp] = imgTemp_np_tumor_swap

        # plt.figure()
        # plt.imshow(np.squeeze(patientStack_kidney_swap[:,:,5]))
        # plt.figure()
        # plt.imshow(imgTemp_np_kidney)

    # STAPLE
    [apparent3M_label1,staple3M_label1,reliability3M_label1] = calConsensus_standardalone(dataPerPatient_kidney)
    mask1 = np.uint8(staple3M_label1 >= kidneyWeight)
    kidneySum.append(np.sum(mask1[:]))

    [apparent3M_label2, staple3M_label2, reliability3M_label2] = calConsensus_standardalone(dataPerPatient_tumor)
    mask2 = np.uint8((staple3M_label2 >= targetWeight)*2.0)
    tumorSum.append(np.sum(mask2[:]))

    labelFinal = mask1+mask2
    labelFinal[labelFinal>2]=2

    patientIndex = testFolderName
    szSlice = np.shape(labelFinal)[2]
    patientFolderPath = os.path.join(savePath, patientIndex)
    try:
        os.stat(patientFolderPath)
    except:
        os.mkdir(patientFolderPath)

    for iter100 in range(0, szSlice):
        tempSaveImg = np.squeeze(labelFinal[:,:,iter100])
        saveFileNameTemp = r"{0:05d}.png".format(iter100+1)
        saveFullPathName = os.path.join(patientFolderPath, saveFileNameTemp)
        cv2.imwrite(saveFullPathName, tempSaveImg)


print("STAPLE process is done...")




