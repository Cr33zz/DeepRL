using System;
using System.Collections.Generic;
using System.Linq;
using DeepQL.MemoryReplays;
using DeepQL.Misc;
using Neuro;
using Neuro.Layers;
using Neuro.Optimizers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQN : ValueFunctionModel
    {
        public DQN(Shape inputShape, int numberOfActions, int[] hiddenLayersNeurons, float learningRate, float discountFactor, int batchSize, BaseExperienceReplay memory)
            : this(inputShape, numberOfActions, learningRate, discountFactor, batchSize, memory)
        {
            ImportanceSamplingWeights = new Tensor(new Shape(1, numberOfActions, 1, batchSize));

            Model = new NeuralNetwork("dqn");
            Model.AddLayer(new Flatten(inputShape));
            for (int i = 0; i < hiddenLayersNeurons.Length; ++i)
                Model.AddLayer(new Dense(Model.LastLayer, hiddenLayersNeurons[i], Activation.ReLU));
            Model.AddLayer(new Dense(Model.LastLayer, numberOfActions, Activation.Linear));
            Model.Optimize(new Adam(learningRate), new CustomHuberLoss(ImportanceSamplingWeights));
        }

        protected DQN(Shape inputShape, int numberOfActions, float learningRate, float discountFactor, int batchSize, BaseExperienceReplay memory)
            : base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            BatchSize = batchSize;
            Memory = memory;
            ErrorChart = new ChartGenerator($"dqn_error", "Q prediction error", "Episode");
            ErrorChart.AddSeries(0, "Abs error", System.Drawing.Color.LightGray);
            ErrorChart.AddSeries(1, $"Avg({ErrorAvg.N}) abs error", System.Drawing.Color.Firebrick);
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
                Memory.Push(new Experience(state, action, reward, nextState, done));

            if (TargetModelUpdateInterval <= 0)
                throw new Exception("Target model update has to be positive.");

            if (TargetModel == null)
                TargetModel = Model.Clone();

            if (!TargetModelUpdateOnEpisodeEnd)
            {
                if (TargetModelUpdateInterval >= 1)
                {
                    if (globalStep % (int)TargetModelUpdateInterval == 0)
                        Model.CopyParametersTo(TargetModel);
                }
                else
                    Model.SoftCopyParametersTo(TargetModel, TargetModelUpdateInterval);
            }
        }

        public override void OnTrain()
        {
            if (Memory.GetSize() >= BatchSize)
                Train(Memory.Sample(BatchSize));
        }

        public override void OnEpisodeEnd(int episode)
        {
            if (TargetModelUpdateOnEpisodeEnd)
                Model.CopyParametersTo(TargetModel);

            if (TrainingsDone > 0)
            {
                ErrorAvg.Add(PerEpisodeErrorAvg);

                ErrorChart.AddData(episode, PerEpisodeErrorAvg, 0);
                ErrorChart.AddData(episode, ErrorAvg.Avg, 1);
                ErrorChart.Save();

                PerEpisodeErrorAvg = 0;
                TrainingsDone = 0;
            }
        }

        public override void SaveState(string filename)
        {
            Model.SaveStateXml(filename);
        }

        public override void LoadState(string filename)
        {
            Model.LoadStateXml(filename);
        }

        protected void Train(List<Experience> experiences)
        {
            var e0 = experiences[0];
            var stateShape = e0.State.Shape;
            var actionShape = e0.Action.Shape;

            Tensor statesBatch = new Tensor(new Shape(stateShape.Width, stateShape.Height, stateShape.Depth, experiences.Count));
            Tensor nextStatesBatch = new Tensor(statesBatch.Shape);

            for (int i = 0; i < experiences.Count; ++i)
            {
                var e = experiences[i];
                e.State.CopyBatchTo(0, i, statesBatch);
                e.NextState.CopyBatchTo(0, i, nextStatesBatch);
            }

            Tensor rewardsBatch = Model.Predict(statesBatch); // this is our original prediction
            Tensor futureRewardsBatch = EnableDoubleDQN ? Model.Predict(nextStatesBatch) : null;
            Tensor futureTargetRewardsBatch = TargetModel.Predict(nextStatesBatch);

            List<float> absErrors = new List<float>();
            ImportanceSamplingWeights.Zero();

            for (int i = 0; i < experiences.Count; ++i)
            {
                var e = experiences[i];

                float futureReward = 0;

                if (EnableDoubleDQN)
                {
                    var nextBestAction = futureRewardsBatch.ArgMax(i);
                    futureReward = futureTargetRewardsBatch[0, nextBestAction, 0, i];
                }
                else
                {
                    futureReward = futureTargetRewardsBatch.Max(i);
                }

                var estimatedReward = e.Reward;
                if (!e.Done)
                    estimatedReward += DiscountFactor * futureReward;

                float error = estimatedReward - rewardsBatch[0, (int)e.Action[0], 0, i];
                absErrors.Add(Math.Abs(error));

                rewardsBatch[0, (int)e.Action[0], 0, i] = estimatedReward;
                ImportanceSamplingWeights[0, (int)e.Action[0], 0, i] = e.ImportanceSamplingWeight;
            }

            Memory.Update(experiences, absErrors);

            var avgError = absErrors.Sum() / experiences.Count;
            ++TrainingsDone;
            PerEpisodeErrorAvg += (avgError - PerEpisodeErrorAvg) / TrainingsDone;

            Model.Fit(statesBatch, rewardsBatch, -1, TrainingEpochs, 0, Track.Nothing);
        }

        public override string GetParametersDescription()
        {
            List<int> hiddenInputs = new List<int>();
            for (int i = 2; i < Model.LayersCount; ++i)
                hiddenInputs.Add(Model.Layer(i).InputShape.Length);

            return $"{base.GetParametersDescription()} batch_size={BatchSize} arch={string.Join("|", hiddenInputs)} target_upd_int={TargetModelUpdateInterval} double_dqn={EnableDoubleDQN} dueling_dqn={EnableDuelingDQN} train_epoch={TrainingEpochs} memory_int={MemoryInterval} target_upd_on_ep_end={TargetModelUpdateOnEpisodeEnd}\n{Memory.GetParametersDescription()}";
        }

        public int BatchSize;
        public int TrainingEpochs = 1;
        // When interval is within (0,1) range, every step soft parameters copy will be performed, otherwise parameters will be copied every interval steps
        public float TargetModelUpdateInterval = 1;
        public bool TargetModelUpdateOnEpisodeEnd = false;
        public bool EnableDoubleDQN = false;
        public bool EnableDuelingDQN = false;
        public int MemoryInterval = 1;
        // Training loss will be clipped to [-DeltaClip, DeltaClip]
        //public float DeltaClip = float.PositiveInfinity;
        //public int ChartSaveInterval = 200;
        protected NeuralNetwork Model;
        protected NeuralNetwork TargetModel;
        protected BaseExperienceReplay Memory;

        private readonly ChartGenerator ErrorChart;
        private float PerEpisodeErrorAvg;
        private readonly MovingAverage ErrorAvg = new MovingAverage(20);
        private int TrainingsDone;
        private Tensor ImportanceSamplingWeights;
    }

    internal class CustomHuberLoss : Huber
    {
        public CustomHuberLoss(Tensor importanceSamplingWeights)
            : base(1)
        {
            ImportanceSamplingWeights = importanceSamplingWeights;
        }

        public override void Derivative(Tensor targetOutput, Tensor output, Tensor result)
        {
            base.Derivative(targetOutput, output, result);
            result.MulElem(ImportanceSamplingWeights, result);
        }

        private Tensor ImportanceSamplingWeights;
    }

}
