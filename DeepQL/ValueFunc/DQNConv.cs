﻿using System.Collections.Generic;
using System.Linq;
using Neuro;
using Neuro.Layers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQNConv : DQN
    {
        public DQNConv(Shape inputShape, int numberOfActions, float learningRate, float discountFactor, int temporalDataSize = 4)
            :base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            Model = new NeuralNetwork("DQNConv");
            Model.AddLayer(new Convolution(inputShape, 8, 32, 2, Activation.ELU));
            Model.AddLayer(new Convolution(Model.LastLayer, 4, 64, 2, Activation.ELU));
            Model.AddLayer(new Convolution(Model.LastLayer, 4, 128, 2, Activation.ELU));
            Model.AddLayer(new Flatten(Model.LastLayer));
            Model.AddLayer(new Dense(Model.LastLayer, 512, Activation.ELU));
            Model.AddLayer(new Dense(Model.LastLayer, numberOfActions, Activation.Softmax));

            TemporalDataSize = temporalDataSize;
        }

        public override void OnStep(Tensor state, Tensor action, float reward, Tensor nextState, bool done)
        {
            UpdateTemporalData(state);
            //Memory.Push(new Transition(Tensor.Merge(TemporalData, 4), action, reward, nextState, done));
        }

        protected override void Train(List<Transition> transitions)
        {
            // make an input batch from transitions and perform a back propagation step

        }

        private void UpdateTemporalData(Tensor state)
        {
            if (TemporalData.Count < TemporalDataSize)
            {
                while (TemporalData.Count < TemporalDataSize)
                    TemporalData.Add(state);
                return;
            }

            TemporalData.Add(state);
            TemporalData.RemoveAt(0);
        }

        private int TemporalDataSize;
        private List<Tensor> TemporalData = new List<Tensor>();
    }
}
