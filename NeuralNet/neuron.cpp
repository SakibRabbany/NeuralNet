//
//  neuron.cpp
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-15.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include "neuron.h"


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0 ; c < numOutputs ; c++) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

double Neuron::randomWeight()
{
    return rand() / double(RAND_MAX);
}

void Neuron::setOutputVal(double value)
{
    m_outputVal = value;
}

double Neuron::getOutputVal() const
{
    return m_outputVal;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    
    //sum the previous layers outputs, this neurons input
    //include the bias neuron from prevlayer
    
    for (unsigned n = 0 ; n < prevLayer.size() ; n++) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double val)
{
    //tanh - range -1..0..1..0
    return tanh(val);
}

double Neuron::transferFunctionDerivative(double val)
{
    //tanh derivative
    return 1 - (val * val);
}

void Neuron::calcOutputGradients (double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    for (unsigned n = 0 ; n < int(nextLayer.size() - 1) ; n++ ) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    //the weights to be updated are in the connection container
    //in the neurons in the preceding layer
    for (unsigned n = 0 ; n < prevLayer.size() ; n++) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight =
                //individual input, magnified by the gradient and the train rate
                eta
                * neuron.getOutputVal()
                * m_gradient
                //also add momentun = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}




