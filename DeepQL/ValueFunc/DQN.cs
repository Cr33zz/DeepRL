using System;
using System.Collections.Generic;
using DeepQL.Misc;
using Neuro;
using Neuro.Layers;
using Neuro.Optimizers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQN : ValueFunctionModel
    {
        public DQN(Shape inputShape, int numberOfActions, int[] hiddenLayersNeurons, float learningRate, float discountFactor, int replaySize)
            : this(inputShape, numberOfActions, learningRate, discountFactor, replaySize)
        {
            Model = new NeuralNetwork("DQN_agent");
            Model.AddLayer(new Flatten(inputShape));
            for (int i = 0; i < hiddenLayersNeurons.Length; ++i)
                Model.AddLayer(new Dense(Model.LastLayer, hiddenLayersNeurons[i], Activation.ReLU));
            Model.AddLayer(new Dense(Model.LastLayer, numberOfActions, Activation.Linear));
            Model.Optimize(new Adam(learningRate), Loss.Huber1);
        }

        protected DQN(Shape inputShape, int numberOfActions, float learningRate, float discountFactor, int replaySize)
            : base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            ReplayMem = new ReplayMemory(replaySize);

            ErrorChart = new ChartGenerator($"DQN_agent_error", "Q prediction error", "Epoch");
            ErrorChart.AddSeries(0, "MSE", System.Drawing.Color.LightGray);
            ErrorChart.AddSeries(1, $"Avg({ErrorAvg.N}) MSE", System.Drawing.Color.Firebrick);
        }

        public override Tensor GetOptimalAction(Tensor state)
        {
            var qValues = Model.Predict(state);
            var action = new Tensor(new Shape(1));
            action[0] = qValues.ArgMax();
            return action;
        }

        public override void OnStep(int step, int globalStep, Tensor state, Tensor action, float reward, Tensor nextState, bool done)
        {
            if (globalStep % MemoryInterval == 0)
                ReplayMem.Push(new Transition(state, action, reward, nextState, done));

            if (UsingTargetModel && (globalStep % TargetModelUpdateInterval == 0))
            {
                if (TargetModel == null)
                    TargetModel = Model.Clone();

                Model.CopyParametersTo(TargetModel);
            }
        }

        public override void OnTrain()
        {
            if (ReplayMem.StorageSize >= BatchSize)
                Train(ReplayMem.Sample(BatchSize));
        }

        public override void OnEpisodeEnd(int episode)
        {
        }

        public override void SaveState(string filename)
        {
            Model.SaveStateXml(filename);
        }

        public override void LoadState(string filename)
        {
            Model.LoadStateXml(filename);
        }

        protected void Train(List<Transition> transitions)
        {
            var stateShape = Model.Layer(0).InputShape;
            Tensor states = new Tensor(new Shape(stateShape.Width, stateShape.Height, stateShape.Depth, transitions.Count));
            Tensor nextStates = new Tensor(states.Shape);

            for (int i = 0; i < transitions.Count; ++i)
            {
                transitions[i].State.CopyBatchTo(0, i, states);
                transitions[i].NextState.CopyBatchTo(0, i, nextStates);
            }

            Tensor rewards = Model.Predict(states); // this is our original prediction
            Tensor futureRewards = (UsingTargetModel ? TargetModel : Model).Predict(nextStates);

            float totalSquareError = 0;

            for (int i = 0; i < transitions.Count; ++i)
            {
                var trans = transitions[i];

                var reward = trans.Reward;
                if (!trans.Done)
                    reward += DiscountFactor * futureRewards.Max(i); // this is the expected prediction for selected action

                float error = reward - rewards[0, (int)trans.Action[0], 0, i];
                totalSquareError += error * error;

                rewards[0, (int)trans.Action[0], 0, i] = reward;
            }

            if (ChartSaveInterval > 0)
            {
                var meanSquareError = totalSquareError / transitions.Count;
                ErrorAvg.Add(meanSquareError);
                ErrorChart.AddData(TrainingsStep, meanSquareError, 0);
                ErrorChart.AddData(TrainingsStep, ErrorAvg.Avg, 1);
                if (TrainingsStep % ChartSaveInterval == 0)
                    ErrorChart.Save();
            }
            ++TrainingsStep;

            Model.Fit(states, rewards, -1, TrainingEpochs, 0, Track.Nothing);
        }

        public override string GetParametersDescription() { return $"{base.GetParametersDescription()} batch_size={BatchSize}"; }

        protected bool UsingTargetModel { get { return TargetModelUpdateInterval > 0; } }

        public int BatchSize = 32;
        public int TrainingEpochs = 1;
        public int TargetModelUpdateInterval = 0;
        public int MemoryInterval = 1;
        // Training loss will be clipped to [-DeltaClip, DeltaClip]
        public float DeltaClip = float.PositiveInfinity;
        public int ChartSaveInterval = 200;
        protected NeuralNetwork Model;
        protected NeuralNetwork TargetModel;
        protected ReplayMemory ReplayMem;
        private readonly ChartGenerator ErrorChart;
        private readonly MovingAverage ErrorAvg = new MovingAverage(100);
        private int TrainingsStep;
    }
}
