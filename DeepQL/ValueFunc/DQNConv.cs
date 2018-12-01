using System.Collections.Generic;
using System.Linq;
using Neuro;
using Neuro.Layers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQNConv : DQN
    {
        public DQNConv(Shape inputShape, int numberOfActions, float learningRate, float discountFactor)
            :base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            Model = new NeuralNetwork("DQNConv");
            Model.AddLayer(new Convolution(inputShape, 8, 32, 2, Activation.ELU));
            Model.AddLayer(new Convolution(Model.LastLayer, 4, 64, 2, Activation.ELU));
            Model.AddLayer(new Convolution(Model.LastLayer, 4, 128, 2, Activation.ELU));
            Model.AddLayer(new Flatten(Model.LastLayer));
            Model.AddLayer(new Dense(Model.LastLayer, 512, Activation.ELU));
            Model.AddLayer(new Dense(Model.LastLayer, numberOfActions, Activation.Softmax));
        }

        public override void OnStep(int step, int globalStep, Tensor state, Tensor action, float reward, Tensor nextState, bool done)
        {
            UpdateTemporalData(state);
            ReplayMem.Push(new Transition(Tensor.Merge(TemporalData, 4), action, reward, nextState, done));
        }

        //protected override void Train(List<Transition> transitions)
        //{
        //    // make an input batch from transitions and perform a back propagation step
        //}

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

        public int TemporalDataSize = 4;
        private List<Tensor> TemporalData = new List<Tensor>();
    }
}
