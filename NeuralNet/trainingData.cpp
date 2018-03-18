//
//  trainingData.cpp
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-18.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#include <stdio.h>
#include "trainingData.h"

using namespace std;

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line, label;
    
    getline(m_trainingDataFile, line);
    
    stringstream ss(line);
    ss >> label;
    if(this->isEof() || label.compare("topology:") != 0){
        abort();
    }
    while(!ss.eof()){
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
    inputVals.clear();
    
    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    
    string label;
    ss >> label;
    
    if(label.compare("in:") == 0){
        double oneValue;
        while(ss >> oneValue){
            inputVals.push_back(oneValue);
        }
    }
    return unsigned(inputVals.size());
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targerOutputVals)
{
    targerOutputVals.clear();
    
    string line, label;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
    
    ss >> label;
    if(label.compare("out:") == 0){
        double oneValue;
        while(ss >> oneValue){
            targerOutputVals.push_back(oneValue);
        }
    }
    
    return unsigned(targerOutputVals.size());
}


bool TrainingData::isEof()
{
    return m_trainingDataFile.eof();
}




