import numpy as np
import torch
import random
import time
import pdb
from utilsADCN import meanStdCalculator, plotPerformance, reduceLabeledData
from model import cluster
from sklearn.metrics import precision_score, normalized_mutual_info_score, adjusted_rand_score, recall_score, f1_score
import progressbar

# ============================= Multi Task Learning =============================
def ADCNmainMT(ADCNnet, dataStreams, nLabeled = 1, layerGrowing = True, nodeEvolution = True, clusterGrowing = True, lwfLoss = True, clusteringLoss = True,
               trainingBatchSize = 16, noOfEpoch = 1, device = torch.device('cpu')):
    # for multi task learning
    # random seed control
    # np.random.seed(0)
    # torch.manual_seed(0)
    # random.seed(0)
    
    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testingTime  = []
    trainingTime = []
    # testingLoss  = []

    prevBatchData = []
    
    # multi task
    currTask    = 0
    prevTask    = 0
    postTaskAcc = []
    preTaskAcc  = []
    memoryData  = torch.Tensor().float()
    nMemory     = []
    nMemoryReplayed = []
    portionMemoryReplayed = []
    
    Y_pred = []
    Y_true = []
    Iter   = []

    # for figure
    AccuracyHistory     = []
    nHiddenLayerHistory = []
    nHiddenNodeHistory  = []
    nClusterHistory     = []
    
    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode  = []
    nHiddenLayer = []
    nCluster     = []
    layerCount   = 0
    
    # initiate network to handle the new task, trained on the initial data in the current task
    start_initialization_train = time.time()
    ADCNnet.initialization(dataStreams.labeledData[currTask], dataStreams.memoryData, layerCount, 
                            batchSize = trainingBatchSize, device = device)
    # pdb.set_trace()

    end_initialization_train = time.time()
    initialization_time      = end_initialization_train - start_initialization_train

    # collection of labeled data
    if nLabeled == 1:
        labeledData  = dataStreams.labeledData[currTask]
        labeledLabel = dataStreams.labeledLabel[currTask]
    elif nLabeled < 1:
        # reduced labeled data
        labeledData, labeledLabel = reduceLabeledData(dataStreams.labeledData[currTask].clone(), 
                                                        dataStreams.labeledLabel[currTask].clone(), nLabeled)
        print('Number of initial allegiance data: ', labeledData.shape[0])
    

    for i in range(len(ADCNnet.hiddenNodeHist)):
        Iter.append(i)
        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        AccuracyHistory.append(0)

    nHiddenNodeHistory = ADCNnet.hiddenNodeHist
    nClusterHistory    = ADCNnet.clusterHistory

    ## batch loop, handling unlabeled samples
    # training is conducted with single epoch. The initialization of a new layer uses epoch.
    # bar = progressbar.ProgressBar(max_value=dataStreams.nBatch)
    # bar = progressbar.ProgressBar()
    
    batchIdx = 0
    for iBatch in range(0, dataStreams.nBatch):
        currTask = dataStreams.taskIndicator[iBatch]

        # update
        start_train = time.time()

        if currTask != prevTask and currTask > prevTask:
            batchIdx = 0
            # memory management
            # pdb.set_trace()
            dataStreams.memoryData = torch.cat((dataStreams.memoryData,memoryData),dim=0)
            memoryData = torch.Tensor().float()

            # test on the prev task before entering curr task. For calculating BWT. 
            prevBatchData  = dataStreams.unlabeledDataTest[prevTask]
            prevBatchLabel = dataStreams.unlabeledLabelTest[prevTask]

            ADCNnet.testing(prevBatchData, prevBatchLabel)

            postTaskAcc.append(ADCNnet.accuracy)

            # test on the current task after finishing prev task. For calculating FWT.
            currBatchData  = dataStreams.unlabeledDataTest[currTask]
            currBatchLabel = dataStreams.unlabeledLabelTest[currTask]

            # update allegiance
            ADCNnet.updateAllegiance(labeledData, labeledLabel, randomTesting = True)

            ADCNnet.testing(currBatchData, currBatchLabel)

            preTaskAcc.append(ADCNnet.accuracy)

            # calculate sample importance
            importantMemories = ADCNnet.sampleSelection(dataStreams.memoryData)

            # initiate network to handle the new task, trained on the initial data in the current task
            ADCNnet.fit(dataStreams.labeledData[currTask], importantMemories, epoch = 50)
            ADCNnet.newMemoryIdx = []     # clear list of important samples
            # pdb.set_trace()

            # augment the collection of unlabeled samples ***************
            if nLabeled == 1:
                labeledData  = torch.cat((labeledData,dataStreams.labeledData[currTask]),0)
                labeledLabel = torch.cat((labeledLabel,dataStreams.labeledLabel[currTask]),0)
            elif nLabeled < 1:
                reducedData, reducedLabel = reduceLabeledData(dataStreams.labeledData[currTask].clone(), 
                                                        dataStreams.labeledLabel[currTask].clone(), nLabeled)
                labeledData  = torch.cat((labeledData,reducedData),0)
                labeledLabel = torch.cat((labeledLabel,reducedLabel),0)
                # print('Number of newly added allegiance data: ', reducedData.shape[0])

        # load data
        batchIdx   = batchIdx + 1
        batchData  = dataStreams.unlabeledData[currTask][(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.unlabeledLabel[currTask][(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]

        

        if iBatch > 0 and layerGrowing:
            # if batchData.shape[0] == 0:
            #     continue

            # drift detection
            ADCNnet.driftDetection(batchData, previousBatchData)

            if ADCNnet.driftStatus == 2:
                # grow layer if drift is confirmed driftStatus == 2
                ADCNnet.layerGrowing()
                layerCount += 1

                # initialization phase
                # need to augment data from previous task
                if currTask > 0:
                    ADCNnet.initialization(dataStreams.labeledData[currTask], dataStreams.memoryData, layerCount, 
                                        batchSize = trainingBatchSize, device = device)
                    # pdb.set_trace()
                else:
                    # in the first task, it is not required to replay any memory
                    memory = torch.Tensor().float()
                    ADCNnet.initialization(dataStreams.labeledData[currTask], memory, layerCount, 
                                        batchSize = trainingBatchSize, device = device)
                    # pdb.set_trace()

        # training data preparation
        previousBatchData = batchData.clone()
        batchData, batchLabel = ADCNnet.trainingDataPreparation(batchData, batchLabel)

        # training
        if ADCNnet.driftStatus == 0 or ADCNnet.driftStatus == 2:  # only train if it is stable or drift
            if currTask > 0:
                # calculate sample importance
                importantMemories = ADCNnet.sampleSelection(dataStreams.memoryData)

                nMemoryReplayed.append(importantMemories.shape[0])
                portionMemoryReplayed.append(importantMemories.shape[0]/dataStreams.memoryData.shape[0])

                # multi task training
                ADCNnet.fit(batchData, importantMemories, epoch = noOfEpoch)
                # pdb.set_trace()
            else:
                # in the first task, it is not required to replay any memory
                memory = torch.Tensor().float()
                ADCNnet.fit(batchData, memory, epoch = noOfEpoch)

                nMemoryReplayed.append(0.0)
                portionMemoryReplayed.append(0.0)
                # pdb.set_trace() 

            ADCNnet.updateNetProperties()

            # add important samples to memory
            memoryData = torch.cat((memoryData,batchData[ADCNnet.newMemoryIdx]),dim=0)
            ADCNnet.newMemoryIdx   = []     # clear list of important samples
            nMemory.append(dataStreams.memoryData.shape[0]+memoryData.shape[0])

            # update allegiance
            ADCNnet.updateAllegiance(labeledData, labeledLabel)

        end_train     = time.time()
        training_time = end_train - start_train

        # testing
        ADCNnet.testing(batchData, batchLabel)
        # if iBatch > 0:
        Y_pred = Y_pred + ADCNnet.predictedLabel.tolist()
        Y_true = Y_true + ADCNnet.trueClassLabel.tolist()

        prevTask = dataStreams.taskIndicator[iBatch]

        Accuracy.append(ADCNnet.accuracy)
        AccuracyHistory.append(ADCNnet.accuracy)
        testingTime.append(ADCNnet.testingTime)
        trainingTime.append(training_time)

        # calculate performance
        if batchIdx%5 == 0 or batchIdx == 0 or batchIdx == 1:
            print(batchIdx,'-th batch',currTask,'-th task')
            print('Accuracy: ',np.mean(Accuracy))
        # testingLoss.append(ADCNnet.testingLoss)
        
        # calculate network evolution
        nHiddenLayer.append(ADCNnet.nHiddenLayer)
        nHiddenNode.append(ADCNnet.nHiddenNode)
        nCluster.append(ADCNnet.nCluster)

        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        nHiddenNodeHistory.append(ADCNnet.nHiddenNode)
        nClusterHistory.append(ADCNnet.nCluster)

        Iter.append(iBatch + i + 1)

        # bar.update(iBatch+1)
    
    
    # final test, all tasks, except the last task. For calculating BWT
    allTaskAccuracies = []
    Y_predTasks       = []
    Y_trueTasks       = []
    for iTask in range(len(dataStreams.unlabeledData)-1):
        ADCNnet.testing(dataStreams.unlabeledDataTest[iTask], dataStreams.unlabeledLabelTest[iTask])
        allTaskAccuracies.append(ADCNnet.accuracy)

        Y_predTasks = Y_predTasks + ADCNnet.predictedLabel.tolist()
        Y_trueTasks = Y_trueTasks + ADCNnet.trueClassLabel.tolist()

    BWT = 1/(dataStreams.nTask-1)*(np.sum(allTaskAccuracies)-np.sum(postTaskAcc))

    # test on the last task
    ADCNnet.testing(dataStreams.unlabeledDataTest[len(dataStreams.unlabeledData)-1], 
                        dataStreams.unlabeledLabelTest[len(dataStreams.unlabeledData)-1])
    allTaskAccuracies.append(ADCNnet.accuracy)

    # test with random initialization. For calculating FWT.
    b_matrix = []
    
    for iTask in range(1, len(dataStreams.unlabeledData)):
        ADCNnet.randomTesting(dataStreams.unlabeledDataTest[iTask], dataStreams.unlabeledLabelTest[iTask])
        b_matrix.append(ADCNnet.accuracy)

    FWT = 1/(dataStreams.nTask-1)*(np.sum(preTaskAcc)-np.sum(b_matrix))

    print('\n')
    print('=== Performance result ===')
    print('Prequential Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Prequential F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Prequential ARI: ',adjusted_rand_score(Y_true, Y_pred))
    print('Prequential NMI: ',normalized_mutual_info_score(Y_true, Y_pred))
    print('Mean Task Accuracy: ',np.mean(allTaskAccuracies),'(+/-)',np.std(allTaskAccuracies))
    print('All Task Accuracy: ',allTaskAccuracies)
    print('Post Task Accuracy: ',postTaskAcc)       # test results on the prev task before entering curr task.
    print('Pre Task Accuracy: ',preTaskAcc)         # test results on the current task after finishing prev task.
    print('B Matrix: ',b_matrix)         # test results on the current task after finishing prev task.
    # print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    # print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    # print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('BWT: ',BWT)
    print('FWT: ',FWT)
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime) + initialization_time,'(+/-)',np.std(trainingTime))
    # print('Testing Loss: ',np.mean(testingLoss),'(+/-)',np.std(testingLoss))
    
    print('\n')
    print('=== Average network evolution ===')
    print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    print('Number of cluster: ',np.mean(nCluster),'(+/-)',np.std(nCluster))
    print('Number of replayed memories: ',np.mean(nMemoryReplayed),'(+/-)',np.std(nMemoryReplayed))
    print('Portion of replayed memories: ',np.mean(portionMemoryReplayed),'(+/-)',np.std(portionMemoryReplayed))

    print('\n')
    print('=== Final network structure ===')
    ADCNnet.getNetProperties()

    # 0: accuracy
    # 1: all tasks accuracy
    # 2: BWT
    # 3: FWT
    # 4: ARI
    # 5: NMI
    # 6: f1_score
    # 7: precision_score
    # 8: recall_score
    # 9: training_time
    # 10: testingTime
    # 11: nHiddenLayer
    # 12: nHiddenNode
    # 13: nCluster
    # 14: nMemory

    allPerformance = [np.mean(Accuracy), np.mean(allTaskAccuracies), BWT, FWT,
                        adjusted_rand_score(Y_true, Y_pred),normalized_mutual_info_score(Y_true, Y_pred),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime) + initialization_time),np.mean(testingTime),
                        ADCNnet.nHiddenLayer,ADCNnet.nHiddenNode,ADCNnet.nCluster,importantMemories.shape[0]]
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    performanceHistory = [Iter,AccuracyHistory,nHiddenLayerHistory,nHiddenNodeHistory,nClusterHistory,nMemory,nMemoryReplayed,portionMemoryReplayed]

    return ADCNnet, performanceHistory, allPerformance