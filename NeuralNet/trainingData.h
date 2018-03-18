//
//  trainingData.h
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-18.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#ifndef trainingData_h
#define trainingData_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

class TrainingData
{
private:
    std::ifstream m_trainingDataFile;
    
public:
    TrainingData(const std::string filename);
    bool isEof();
    void getTopology(std::vector<unsigned> &topology);
    
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targerOutputVals);
    
};



#endif /* trainingData_h */
