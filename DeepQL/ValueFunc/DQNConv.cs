using System.Collections.Generic;
using System.Data.Common;
using DeepQL.MemoryReplays;
using Neuro;
using Neuro.Layers;
using Neuro.Models;
using Neuro.Optimizers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQNConv : DQN
    {
        public DQNConv(int numberOfActions, float learningRate, float discountFactor, int batchSize, BaseExperienceReplay memory)
            :base(null, numberOfActions, learningRate, discountFactor, batchSize, memory)
        {
			Tensor.SetOpMode(Tensor.OpMode.GPU);

			Shape inputShape = new Shape(InputHeight, InputHeight, TemporalDataSize);

            Net = new NeuralNetwork("DQNConv");
            var Model = new Sequential();
			Model.AddLayer(new Convolution(inputShape, 8, 32, 2, Activation.ELU));
			Model.AddLayer(new Convolution(Model.LastLayer, 4, 64, 2, Activation.ELU));
			Model.AddLayer(new Convolution(Model.LastLayer, 4, 128, 2, Activation.ELU));
			Model.AddLayer(new Flatten(Model.LastLayer));
			Model.AddLayer(new Dense(Model.LastLayer, 512, Activation.ELU));
			Model.AddLayer(new Dense(Model.LastLayer, numberOfActions, Activation.Softmax));
			Net.Model = Model;
			Net.Optimize(new Adam(learningRate), new CustomHuberLoss(ImportanceSamplingWeights));

		}

		public override void OnStep(int step, int globalStep, Tensor state, Tensor action, float reward, Tensor nextState, bool done)
		{
			var nextStateScaled = RescaleState(state, InputWidth, InputHeight);
			var tempState = LastTemporalState;
			UpdateTemporalData(nextStateScaled);
			var nextTempState = Tensor.MergeIntoDepth(TemporalData);
			if (TemporalData.Count >= TemporalDataSize && tempState != null)
				base.OnStep(step, globalStep, tempState, action, reward, nextTempState, done);
			LastTemporalState = nextTempState;
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

        private Tensor RescaleState(Tensor state, int newWidth, int newHeight)
        {
	        float scaleWidth = newWidth / (float)state.Width;
	        float scaleHeight = newHeight / (float)state.Height;
			var result = new Tensor(new Shape(newWidth, newHeight));
			for (int y = 0; y < newHeight; ++y)
			for (int x = 0; x < newWidth; ++x)
			{
				result[x, y] = state[(int)(x / scaleWidth), (int)(y / scaleHeight)];
			}

			return result;
        }

        public const int InputWidth = 84;
        public const int InputHeight = 84;
		public const int TemporalDataSize = 4;
		private Tensor LastTemporalState;
        private List<Tensor> TemporalData = new List<Tensor>();
    }
}
