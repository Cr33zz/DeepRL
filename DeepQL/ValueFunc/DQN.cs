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
        public DQN(Shape inputShape, int numberOfActions, double learningRate, double discountFactor, int replaySize = 1000)
            : base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            Model = new NeuralNetwork("DQN_agent");
            Model.AddLayer(new Flatten(inputShape));
            Model.AddLayer(new Dense(Model.LastLayer, 24, Activation.ReLU));
            Model.AddLayer(new Dense(Model.LastLayer, 24, Activation.ReLU));
            Model.AddLayer(new Dense(Model.LastLayer, numberOfActions, Activation.Linear));
            Model.Optimize(new Adam(learningRate), Loss.MeanSquareError);

            TargetModel = Model.Clone();

            ReplayMem = new ReplayMemory(replaySize);

            ErrorChart = new ChartGenerator($"DQN_agent_error", "Reward prediction error", "Episode");
            ErrorChart.AddSeries(0, "Error", System.Drawing.Color.LightGray);
            ErrorChart.AddSeries(1, "Avg Error", System.Drawing.Color.Red);
        }

        public override Tensor GetOptimalAction(Tensor state)
        {
            var qValues = Model.Predict(state);
            var action = new Tensor(new Shape(1));
            action[0] = qValues.ArgMax();
            return action;
        }

        public override void OnStep(Tensor state, Tensor action, double reward, Tensor nextState, bool done)
        {
            ReplayMem.Push(new Transition(state, action, reward, nextState, done));

            if (ReplayMem.StorageSize >= BatchSize)
                Train(ReplayMem.Sample(BatchSize));
        }

        public override void OnEpisodeEnd(int episode)
        {
            Model.CopyParametersTo(TargetModel);
        }

        public override void SaveState(string filename)
        {
            Model.SaveStateXml(filename);
        }

        public override void LoadState(string filename)
        {
            Model.LoadStateXml(filename);
        }

        protected override void Train(List<Transition> transitions)
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
            Tensor futureRewards = TargetModel.Predict(nextStates);

            double totalAbsError = 0;

            for (int i = 0; i < transitions.Count; ++i)
            {
                var trans = transitions[i];

                var reward = trans.Reward;
                if (!trans.Done)
                    reward += DiscountFactor * futureRewards.Max(i); // this is the expected prediction for selected action

                totalAbsError += Math.Abs(rewards[0, (int)trans.Action[0], 0, i] - reward);

                rewards[0, (int)trans.Action[0], 0, i] = reward;
            }

            var averageAbsError = totalAbsError / transitions.Count;
            MoveAvg.Add(averageAbsError);
            ErrorChart.AddData(TrainingsCounts, averageAbsError, 0);
            ErrorChart.AddData(TrainingsCounts, MoveAvg.Avg, 1);
            ++TrainingsCounts;
            if (TrainingsCounts % 200 == 0)
                ErrorChart.Save();

            Model.Fit(states, rewards, 1, 0, Track.Nothing);
        }

        public int BatchSize = 32;
        protected NeuralNetwork Model;
        protected NeuralNetwork TargetModel;
        private ReplayMemory ReplayMem;
        private ChartGenerator ErrorChart;
        private MovingAverage MoveAvg = new MovingAverage(100);
        private int TrainingsCounts;
    }
}
