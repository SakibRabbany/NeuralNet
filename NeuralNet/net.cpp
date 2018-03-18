//
//  net.cpp
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-15.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include "net.h"
#include "neuron.h"


using namespace std;


Net::Net(const vector <unsigned> &topology)
{
    int numLayers = int(topology.size());
    for (unsigned layerNum = 0 ; layerNum < numLayers ;  layerNum++) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        // made a new layer, fill layer with neurons and bias neuron
        for (unsigned neuronNum = 0 ; neuronNum <= topology[layerNum] ; neuronNum++) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "made a neuron" << endl;
        }
        
        //force the bias neurons output to 1
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);
    
    //Assign input values to input neurons
    for (unsigned i = 0 ; i < inputVals.size() ; i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //Forward propagate
    for (unsigned layerNum = 1 ; layerNum < m_layers.size() ; layerNum++) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0 ; n < m_layers[layerNum].size() - 1 ; n++) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    //calculate overall net error (rms of output neurons
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for (unsigned n = 0 ; n < outputLayer.size() - 1 ; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
    
    //implement a recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);
    
    //calculate output layer gradients
    for (unsigned n = 0 ; n < outputLayer.size() - 1 ; n++) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    
    //calculate gradients on hidden layers
    for (unsigned layerNum = int(m_layers.size() - 2) ; layerNum > 0; layerNum--) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        
        for (unsigned n = 0 ; n < hiddenLayer.size() ; n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    
    //for all layers from outputs to first hidden layer,
    //update connection weights
    
    for (unsigned layerNum = int(m_layers.size() - 1) ; layerNum > 0 ; layerNum--) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        
        for (unsigned n = 0 ; n < layer.size() - 1 ; n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}


void Net::getResults (vector <double> &resultVals) const
{
    resultVals.clear();
    
    for (unsigned n = 0 ; n < m_layers.back().size() - 1 ; n++) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


double Net::getRecentAveragedError()
{
    return m_recentAverageError;
}
