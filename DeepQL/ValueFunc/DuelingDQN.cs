using System.Collections.Generic;
using DeepQL.MemoryReplays;
using Neuro;
using Neuro.Layers;
using Neuro.Models;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DuelingDQN : DQN
    {
        public DuelingDQN(Shape inputShape, int numberOfActions, int[] hiddenLayersNeurons, float learningRate, float discountFactor, int batchSize, BaseExperienceReplay memory)
            : base(inputShape, numberOfActions, hiddenLayersNeurons, learningRate, discountFactor, batchSize, memory)
        {
            Net = new NeuralNetwork("DuelingDQN");            
            var input = new Flatten(inputShape);
            LayerBase lastLayer = input;
            for (int i = 0; i < hiddenLayersNeurons.Length; ++i)
                lastLayer = new Dense(lastLayer, hiddenLayersNeurons[i], Activation.ReLU);

            LayerBase stateValue = new Dense(lastLayer, 1);
            stateValue = new Lambda(new []{stateValue}, new Shape(1, numberOfActions), (inps, outp) => { outp.Zero();}, (outpG, inpsG) => { });

            var actionAdvantage = new Dense(lastLayer, numberOfActions);

            var output = new Merge(new []{stateValue, actionAdvantage}, Merge.Mode.Sum, Activation.Linear);
            Net.Model = new Flow(new []{input}, new []{output});
        }
    }
}
