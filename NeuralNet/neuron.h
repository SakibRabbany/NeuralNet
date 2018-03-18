//
//  neuron.h
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-15.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#ifndef neuron_h
#define neuron_h

#include <vector>

class Neuron;

typedef std::vector <Neuron> Layer;

struct Connection {
    double weight;
    double deltaWeight;
};


class Neuron
{
private:
    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;
    static double eta;      //[0..1] overall net training rate
    static double alpha;    //[0..1] ultiplier of last weight change (momentum)
    std::vector <Connection> m_outputWeights;
    
    static double randomWeight();
    static double transferFunction(double val);
    static double transferFunctionDerivative(double val);
    double sumDOW(const Layer &nextLayer) const;
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double value);
    double getOutputVal() const;
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
};

#endif /* neuron_h */
