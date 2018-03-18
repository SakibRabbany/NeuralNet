//
//  main.cpp
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-19.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#include <stdio.h>
#include <cassert>
#include "net.h"
#include "neuron.h"
#include "trainingData.h"

using namespace std;


void showVectorVals(string label, vector<double> &vec){
    cout << label << " ";
    for (unsigned i = 0 ; i < vec.size() ; i++) {
        cout << vec[i] << " ";
    }
    
    cout << endl;
}


int main() {
    TrainingData trainData("trainingData.txt");
    
    vector<unsigned> topology;
    trainData.getTopology(topology);
    
    Net myNet(topology);
    
    vector<double> inputVals, targetVals, resultVals;
    
    int trainingPass = 0;
    
    while(!trainData.isEof()){
        ++trainingPass;
        cout << endl << "Pass: " << trainingPass;
        
        if(trainData.getNextInputs(inputVals) != topology[0]){
            break;
        }
        
        showVectorVals(": Inputs: ", inputVals);
        myNet.feedForward(inputVals);
        
        myNet.getResults(resultVals);
        showVectorVals("Outputs: ", resultVals);
        
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets: ", targetVals);
        
        assert(targetVals.size() == topology.back());
        
        myNet.backProp(targetVals);
        cout << "Net recent average error: " << myNet.getRecentAveragedError() << endl;
    }
    cout << "Done" << endl;
    return 0;
}
