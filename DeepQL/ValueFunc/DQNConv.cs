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
        public DQNConv(int[] inputSize, int numberOfActions, float learningRate, float discountFactor, int batchSize, BaseExperienceReplay memory)
            :base(null, numberOfActions, learningRate, discountFactor, batchSize, memory)
        {
			Tensor.SetOpMode(Tensor.OpMode.GPU);

			InputSize = inputSize;
			Shape inputShape = new Shape(inputSize[0], inputSize[1], TemporalDataSize);

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

        public override Tensor GetOptimalAction(Tensor state)
        {
            // we have to use last TemporalDataSize frames (LastTemporalState) instead of single last frame (state)
            var qValues = Net.Predict(LastTemporalState)[0];
            var action = new Tensor(new Shape(1));
            action[0] = qValues.ArgMax();
            return action;
        }

        public override void OnReset(Tensor initialState)
        {
            var initialStateScaled = RescaleState(initialState, InputSize[0], InputSize[1]);
            TemporalData.Clear();
            UpdateTemporalData(initialStateScaled);
            LastTemporalState = Tensor.MergeIntoDepth(TemporalData, TemporalDataSize);
        }

        public override void OnStep(int step, int globalStep, Tensor state, Tensor action, float reward, Tensor nextState, bool done)
		{
			var nextStateScaled = RescaleState(state, InputSize[0], InputSize[1]);
			var tempState = LastTemporalState;
			UpdateTemporalData(nextStateScaled);
			var nextTempState = Tensor.MergeIntoDepth(TemporalData, TemporalDataSize);

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

        public int[] InputSize;
		public const int TemporalDataSize = 4;
		private Tensor LastTemporalState;
        private List<Tensor> TemporalData = new List<Tensor>();
    }
}
