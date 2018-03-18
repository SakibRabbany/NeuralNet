//
//  net.h
//  NeuralNet
//
//  Created by Sakib Rabbany on 2017-06-15.
//  Copyright © 2017 Sakib Rabbany. All rights reserved.
//

#ifndef net_h
#define net_h

#include <vector>

class Neuron;

typedef std::vector <Neuron> Layer;

class Net
{
private:
    std::vector <Layer> m_layers;    // m_layers[layer][neuron]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
public:
    Net (const std::vector <unsigned> &topology);
    void feedForward (const std::vector <double> &inputVals);
    void backProp (const std::vector <double> &targetVals);
    void getResults (std::vector <double> &resultVals) const;
    double getRecentAveragedError();
};



#endif /* net_h */
