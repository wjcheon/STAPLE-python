import numpy as np
import copy
import math
from scipy import special
#import matplotlib.pyplot as plt

def find3d_ind(bigMask_):
    indV = np.argwhere(bigMask_==True)
    indVx = indV[:,0]
    indVy = indV[:, 1]
    indVz = indV[:, 2]

    xMin = np.min(indVx)
    xMax = np.max(indVx)
    yMin = np.min(indVy)
    yMax = np.max(indVy)
    zMin = np.min(indVz)
    zMax = np.max(indVz)
    return xMin, xMax, yMin, yMax, zMin, zMax

def staple_wjcheon(D,iterlim,p,q):
    #---- inputs:
    # *D: a matrix of N(voxels) x R(binary decisions by experts)
    # *p: intial sensitivity
    # *q: intial specificity
    # *iterlim: iteration limit
    #---- outputs:
    # *p: final sensitivity estimate
    # *q: final specificity estimate
    # *W: estimated belief in true segmentation
    [N, R] =np.shape(D)
    Tol=1e-4
    iter=0
    gamma= np.sum(np.sum(D,axis=0)/(R*N))
    W = np.zeros((N,1), dtype=np.single)
    S0 = np.sum(W)

    stapleV = []
    sen = []
    spec = []
    Sall = []
    while(True):
        iter=iter+1
        Sall.append(S0)

        ind1 = np.equal(D,1)
        ind0 = np.equal(D,0)
        ind1_not = np.logical_not(ind1)
        ind0_not = np.logical_not(ind0)

        p = np.repeat(p, N, axis=0)
        p1 = copy.deepcopy(p)
        p0 = copy.deepcopy(1-p1)

        p1[ind1_not]=1
        p0[ind0_not]=1
        a=gamma*np.multiply(np.prod(p1,axis=1),np.prod(p0,axis=1))
        del p1, p0

        q = np.repeat(q, N, axis=0)
        q0 = copy.deepcopy(q)
        q1 = copy.deepcopy(1-q0)
        q1[ind1_not]=1
        q0[ind0_not]=1
        del ind1, ind0, ind1_not, ind0_not
        b= (1-gamma)*np.multiply(np.prod(q0, axis=1), np.prod(q1,axis=1))
        del q1, q0

        W = np.single(np.divide(a, a+b))
        W = np.single(np.reshape(W, (1, len(W))))

        del a, b, p, q

        p= np.divide(np.matmul(W,D),np.sum(W))
        q= np.divide(np.matmul(1-W, 1-D), np.sum(1-W))
        # Check convergence
        S= np.single(np.sum(W))
        if np.abs(S-S0) < Tol:
            print("STAPLE converged in {} iterations".format(iter))
            break
        else:
            S0=np.single(S)

        # Check iteration limit
        if (iter>iterlim):
            print("STAPLE: Number of iterations exceeded without convergence (convergence tolerance = %e)".format(Tol))
            break

    return W, p, q, Sall

def getUniformScanXYZVals_standardalone(rtstStruct):
    sizeArray = np.shape(rtstStruct)
    sizeDim1 = sizeArray[0]-1
    sizeDim2 = sizeArray[1]-1

    xOffset =0
    yOffset =0
    firstZValue =0
    grid2Units = 0.9765625
    grid1Units = 0.9765625
    sliceThickness = 5.0

    #xVals = xOffset - (sizeDim2*grid2Units)/2 : grid2Units :
    xSt = xOffset - (sizeDim2*grid2Units)/2
    xEnd = xOffset + (sizeDim2*grid2Units)/2 + grid2Units
    xVals = np.arange(xSt, xEnd, grid2Units)

    ySt = yOffset - (sizeDim1*grid1Units)/2
    yEnd = yOffset + (sizeDim1*grid1Units)/2 + grid2Units
    yVals = np.arange(ySt, yEnd, grid1Units)
    yVals = np.flip(yVals)

    nZSlices = sizeArray[2];
    zSt = firstZValue
    zEnd = sliceThickness * (nZSlices - 1) + firstZValue + sliceThickness
    zVals = np.arange(zSt, zEnd, sliceThickness)

    return xVals, yVals, zVals

def kappa_stats(D, ncat):
    [N, M] = np.shape(D)
    lk= len(ncat)
    x=[]
    for iterVal in range(0, lk):
        x.append(np.sum(np.equal(D, ncat[iterVal]), axis=1))
    x = np.transpose(x)

    p = np.divide(np.sum(x, axis=0),(N*M))
    eps = np.finfo(float).eps
    k_a = np.sum(np.multiply(x, M-x), axis=0)
    k_b = (N*M*(M-1))*np.multiply(p,(1-p))+eps
    k = 1-np.divide(k_a,k_b)
    sek= np.sqrt(2/(N*M*(M-1)))
    pk = drxlr_get_p_gaussian(np.divide(k,sek))/2
    kappa_a = N*M*M-np.sum(np.sum(np.multiply(x,x)))
    kappa_b = N*M*(M-1)*np.sum(np.multiply(p,(1-p)))+eps
    kappa = 1-(kappa_a/kappa_b)
    sekappa_a = np.sum(np.multiply(p, (1-p))*np.sqrt(N*M*(M-1))+eps)
    sekappa_b = np.power(np.sqrt(np.sum(np.multiply(p, (1-p)))),2) - np.sum(np.multiply(np.multiply(p, 1-p),(1-2*p)))
    sekappa = np.sqrt(2)/sekappa_a*sekappa_b
    z = kappa/sekappa
    pval = drxlr_get_p_gaussian(z)/2


    return kappa, pval, k, pk


def drxlr_get_p_gaussian(x):
    p = special.erfc(np.abs(x)/np.sqrt(2))

    return p

def calConsensus_standardalone(rtstStructs):

    keysDictionary = list(rtstStructs.keys())
    bigMask = rtstStructs.get(keysDictionary[0])
    dictionaryLength = len(rtstStructs)
    for iter1 in range(0, dictionaryLength):
        bigMask = np.logical_or(bigMask, rtstStructs.get(keysDictionary[iter1]))

    iMin, iMax, jMin, jMax, kMin, kMax = find3d_ind(bigMask);

    averageMask3M =np.zeros((iMax-iMin+1, jMax-jMin+1, kMax-kMin+1), dtype=np.single)
    rateMat = []
    for iter1 in range(0, dictionaryLength):
        mask3M = rtstStructs.get(keysDictionary[iter1])
        mask3M_ROI = np.asanyarray(mask3M[iMin:iMax+1, jMin: jMax+1, kMin: kMax+1])
        averageMask3M = averageMask3M + mask3M_ROI
        mask3M_ROI_flat = mask3M_ROI.flatten()
        rateMat.append(mask3M_ROI_flat)
    averageMask3M = averageMask3M / dictionaryLength
    rateMat = np.transpose(rateMat)
    scanNum = 1;
    iterlim = 100;
    senstart = 0.9999 * np.ones((1, dictionaryLength))
    specstart = 0.9999 * np.ones((1, dictionaryLength))
    [stapleV, sen, spec, Sall] = staple_wjcheon(rateMat, iterlim, np.single(senstart), np.single(specstart))

    mean_sen = np.mean(sen)
    std_sen = np.std(sen, ddof=1)
    mean_spec = np.mean(spec)
    std_spec = np.std(spec, ddof=1)

    [xUnifV, yUnifV, zUnifV] = getUniformScanXYZVals_standardalone(mask3M)
    vol = (xUnifV[2] - xUnifV[1]) * (yUnifV[1] - yUnifV[2]) * (zUnifV[2] - zUnifV[1])
    vol = vol* 0.001

    numBins = 20
    obsAgree = np.linspace(0.001, 1, numBins)
    rater_prob = np.mean(rateMat, axis=0)
    chance_prob = np.sqrt(np.multiply(rater_prob, (1-rater_prob)))
    chance_prob = np.reshape(chance_prob, (1, np.shape(chance_prob)[0]))
    chance_prob_mat = np.repeat(chance_prob, np.shape(rateMat)[0], axis=0)
    reliabilityV = np.mean(np.divide((rateMat-chance_prob_mat), (1-chance_prob_mat)), axis=1)
    del rater_prob, chance_prob, chance_prob_mat

    volV = []
    volStapleV =[]
    volKappaV =[]
    for iter10 in range(0, len(obsAgree)):
        updatedValue = np.sum((averageMask3M.flatten()>=obsAgree[iter10])*vol)
        volV.append(updatedValue)
        updatedValue2 = np.sum((stapleV.flatten() >= obsAgree[iter10]) * vol)
        volStapleV.append(updatedValue2)
        updatedValue3 = np.sum((reliabilityV.flatten() >= obsAgree[iter10]) * vol)
        volKappaV.append(updatedValue3)

    # calculate overall kappa
    [kappa, pval, k, pk] = kappa_stats(rateMat, [0, 1])
    min_vol = np.min(np.sum(rateMat, axis=0))*vol
    max_vol = np.max(np.sum(rateMat, axis=0))*vol
    mean_vol = np.mean(np.sum(rateMat, axis=0))*vol
    sd_vol = np.std(np.sum(rateMat, axis=0), ddof=1)*vol

    print('-------------------------------------------')
    print('Overall kappa: {0:1.8f}'.format(kappa))
    print('p-value: {0:1.8f}'.format(pval))
    print('Mean Sensitivity: {0:1.8f}'.format(mean_sen))
    print('Std. Sensitivity: {0:1.8f}'.format(std_sen))
    print('Mean Specificity: {0:1.8f}'.format(mean_spec))
    print('Std. Specificity: {0:1.8f}'.format(std_spec))
    print('Min. volume: {0:1.8f}'.format(min_vol))
    print('Max. volume: {0:1.8f}'.format(max_vol))
    print('Mean volume: {0:1.8f}'.format(mean_vol))
    print('Std. volume: {0:1.8f}'.format(sd_vol))
    print('Intersection volume: {0:1.8f}'.format(volV[-1]))
    print('Union volume: {0:1.8f}'.format(volV[1]))
    print('-------------------------------------------')

    len_x, len_y, len_z = np.shape(averageMask3M)
    stapleV_reshape = np.reshape(stapleV, (len_x, len_y, len_z))
    staple3M = np.zeros_like(bigMask, dtype=np.single)
    staple3M[iMin:iMax+1, jMin: jMax+1, kMin: kMax+1] = stapleV_reshape
    #
    reliabilityV_reshape = np.reshape(reliabilityV, (len_x, len_y, len_z))
    reliability3M = np.zeros_like(bigMask, dtype=np.single)
    reliability3M[iMin:iMax + 1, jMin: jMax + 1, kMin: kMax + 1] = reliabilityV_reshape
    #
    apparent3M =np.zeros_like(bigMask, dtype=np.single)
    apparent3M[iMin:iMax + 1, jMin: jMax + 1, kMin: kMax + 1] = averageMask3M

    # tempImg = aa[:,:,32]
    # plt.figure()
    # plt.imshow(tempImg)

    return apparent3M, staple3M, reliability3M