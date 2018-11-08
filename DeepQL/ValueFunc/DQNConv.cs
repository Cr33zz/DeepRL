using System.Collections.Generic;
using System.Linq;
using Neuro;
using Neuro.Layers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQNConv : ValueFunctionModel
    {
        public DQNConv(Shape inputShape, int numberOfActions, double learningRate, double discountFactor, int temporalDataSize = 4)
            :base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            Net = new NeuralNetwork("DQNConv");
            Net.AddLayer(new Convolution(inputShape, 8, 32, 2, Activation.ELU));
            Net.AddLayer(new Convolution(Net.LastLayer(), 4, 64, 2, Activation.ELU));
            Net.AddLayer(new Convolution(Net.LastLayer(), 4, 128, 2, Activation.ELU));
            Net.AddLayer(new Flatten(Net.LastLayer()));
            Net.AddLayer(new Dense(Net.LastLayer(), 512, Activation.ELU));
            Net.AddLayer(new Dense(Net.LastLayer(), numberOfActions, Activation.Softmax));

            Memory = new ReplayMemory(100);

            TemporalDataSize = temporalDataSize;
        }

        public override void OnTransition(Tensor state, int action, double reward, Tensor nextState)
        {
            UpdateTemporalData(state);
            //Memory.Push(new Transition(Tensor.Merge(TemporalData, 2), action, reward, nextState)); merge over depth
        }

        public override double GetOptimalAction(Tensor state)
        {
            var output = Net.FeedForward(state);
            return output.ArgMax();
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
        private NeuralNetwork Net;
        private ReplayMemory Memory;
    }
}
