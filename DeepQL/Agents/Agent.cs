using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using DeepQL.Environments;
using DeepQL.Misc;
using Neuro.Tensors;

namespace DeepQL.Agents
{
    public enum EEpsilonDecayMode
    {
        EveryStep,
        EveryEpisode,
    }

    public abstract class Agent
    {
        protected Agent(string name, Env env)
        {
            Name = name;
            Env = env;
        }

        public void Train(int episodes, int maxStepsPerEpisode)
        {
            Epsilon = MaxEpsilon;

            var rewardChart = new Neuro.ChartGenerator($"{Name}_reward.png", Name, "Episode");
            rewardChart.AddSeries(0, "Reward", System.Drawing.Color.LightGray);
            rewardChart.AddSeries(1, "Avg reward", System.Drawing.Color.Blue);
            var moveAvg = new MovingAverage(100);

            int globalStep = 0;

            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();

                float totalReward = 0;
                int step = 0;

                for (; step < maxStepsPerEpisode; ++step, ++globalStep)
                {
                    Tensor action;

                    if (GlobalRandom.Rng.NextDouble() < Epsilon)
                        action = Env.ActionSpace.Sample(); // explore
                    else
                        action = GetOptimalAction(); // exploit

                    bool done = Env.Step(action, out var observation, out var reward);

                    if (done && !float.IsNaN(RewardOnDone))
                        reward = RewardOnDone;

                    if (ClipReward)
                        reward = Neuro.Tools.Clip(reward, -1, 1);

                    totalReward += reward;

                    if (TrainingRenderFreq > 0 && (ep % TrainingRenderFreq == 0))
                        RenderEnv();
                    
                    OnStep(step, globalStep, action, reward, observation, done);

                    LastObservation = observation;

                    if (!TrainOnlyOnEpisodeEnd && (globalStep > StepsBeforeTraining))
                        OnTrain();

                    if (EpsilonDecayMode == EEpsilonDecayMode.EveryStep)
                        DecayEpsilon();

                    if (done)
                        break;
                }

                if (TrainOnlyOnEpisodeEnd && (globalStep >= StepsBeforeTraining))
                    OnTrain();

                OnEpisodeEnd(ep);

                moveAvg.Add(totalReward);
                rewardChart.AddData(ep, totalReward, 0);
                rewardChart.AddData(ep, moveAvg.Avg, 1);

                if (SaveFreq > 0 && (ep % SaveFreq == 0))
                    Save($"{Name}_{ep}");

                if (Verbose)
                    LogLine($"Episode: {ep}  reward(avg): {Math.Round(totalReward, 2)}({Math.Round(moveAvg.Avg, 2)})  steps: {step}  epsilon: {Math.Round(Epsilon, 4)}");

                if (EpsilonDecayMode == EEpsilonDecayMode.EveryEpisode)
                    DecayEpsilon();

                if (ep % 20 == 0)
                    rewardChart.Save();
            }

            rewardChart.Save();
            SaveLog();
        }

        public float Test(int episodes, int maxStepsPerEpisode, bool render)
        {
            float[] totalRewards = new float[episodes];

            for (int ep = 0; ep < episodes; ++ep)
            {
                LastObservation = Env.Reset();
                float totalReward = 0;

                for (int step = 0; step < maxStepsPerEpisode; ++step)
                {
                    Tensor action = GetOptimalAction();

                    bool done = Env.Step(action, out var observation, out var reward);

                    LastObservation = observation;
                    totalReward += reward;

                    if (render)
                        RenderEnv();

                    if (done)
                        break;
                }

                totalRewards[ep] = totalReward;

                if (Verbose)
                    LogLine($"Episode# {ep}  reward: {Math.Round(totalReward, 2)}");
            }

            SaveLog();
            return totalRewards.Sum() / episodes;
        }

        public virtual void Save(string filename) { }
        public virtual void Load(string filename) { }

        protected abstract Tensor GetOptimalAction();
        protected abstract void OnStep(int step, int globalStep, Tensor action, float reward, Tensor nextState, bool done);
        protected abstract void OnTrain();
        protected virtual void OnEpisodeEnd(int episode) { }

        private void RenderEnv()
        {
            Env.Render();
            Thread.Sleep(1000 / RenderFreq);
        }

        private void DecayEpsilon()
        {
            Epsilon = Math.Max(MinEpsilon, Epsilon * EpsilonDecay);
        }

        private void LogLine(string text)
        {
            LogLines.Add(text);
            Console.WriteLine(text);
        }

        private void SaveLog()
        {
            File.WriteAllLines($"{Name}_log.txt", LogLines);
        }

        protected Tensor LastObservation;
        protected readonly Env Env;

        public float MinEpsilon = 0.01f;
        public float MaxEpsilon = 1.0f;
        public float EpsilonDecay = 0.995f;
        public EEpsilonDecayMode EpsilonDecayMode = EEpsilonDecayMode.EveryEpisode;
        public bool TrainOnlyOnEpisodeEnd = false;
        // Number of total steps that have to be performed before agent will start training
        public int StepsBeforeTraining = 0;
        // Parameters training parameters will be saved every that number of episodes (0 for no saving)
        public int SaveFreq = 50;
        // Training episode will be rendered every that number of episodes (0 for no rendering)
        public int TrainingRenderFreq = 0;
        // When not NaN, reward for step in which simulation ended will be overriten with that value
        public float RewardOnDone = float.NaN;
        // Used for controling rendering FPS
        public int RenderFreq = 30;
        // When enabled rewards will be clipped to [-1, 1] range (inclusive)
        public bool ClipReward = false;
        public bool Verbose = false;

        protected float Epsilon; // Exploration probability
        private readonly string Name;
        private List<string> LogLines = new List<string>();
    }
}
