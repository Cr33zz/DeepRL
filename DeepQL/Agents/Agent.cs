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

            var rewardChart = new Neuro.ChartGenerator($"{Name}_reward", $"{Name}\n{GetParametersDescription()}", "Episode");
            rewardChart.AddSeries(0, "Reward", System.Drawing.Color.LightGray);
            rewardChart.AddSeries(1, $"Avg({RewardAverageN}) reward", System.Drawing.Color.Blue);
            rewardChart.AddSeries(2, $"Avg({StepsAverageN}) steps per episode\n(right axis)", System.Drawing.Color.CornflowerBlue, true);
            rewardChart.AddSeries(3, "Reward high score", System.Drawing.Color.DarkOrchid);
            var rewardAvg = new MovingAverage(RewardAverageN);
            var stepsAvg = new MovingAverage(StepsAverageN);

            int globalStep = 0;

            for (int ep = 1; ep <= episodes; ++ep)
            {
                LastObservation = Env.Reset();

                float totalReward = 0;
                int step = 0;

                for (; step < maxStepsPerEpisode; ++step, ++globalStep)
                {
                    Tensor action;

                    if (globalStep < WarmupSteps || GlobalRandom.Rng.NextDouble() < Epsilon)
                        action = Env.ActionSpace.Sample(); // explore
                    else
                        action = GetOptimalAction(); // exploit

                    if (globalStep >= WarmupSteps && EpsilonDecayMode == EEpsilonDecayMode.EveryStep)
                        DecayEpsilon();

                    bool done = Env.Step(action, out var observation, out var reward);

                    if (done && !float.IsNaN(RewardOnDone))
                        reward = RewardOnDone;

                    if (RewardClipping)
                        reward = reward > 0 ? 1 : (reward < 0 ? -1 : 0);

                    totalReward += reward;

                    if (TrainRenderInterval > 0 && (ep % TrainRenderInterval == 0))
                        RenderEnv();
                    
                    OnStep(step, globalStep, action, reward, observation, done);

                    LastObservation = observation;

                    if ((globalStep % TrainInterval == 0) && (globalStep >= WarmupSteps))
                        OnTrain();

                    if (done)
                        break;
                }

                OnEpisodeEnd(ep);

                rewardAvg.Add(totalReward);
                stepsAvg.Add(step);
                rewardChart.AddData(ep, totalReward, 0);
                rewardChart.AddData(ep, rewardAvg.Avg, 1);
                rewardChart.AddData(ep, stepsAvg.Avg, 2);

                if (totalReward > RewardHighScore)
                {
                    rewardChart.AddData(ep, totalReward, 3);
                    RewardHighScore = totalReward;
                }

                if (SaveFreq > 0 && (ep % SaveFreq == 0))
                    Save($"{Name}_{ep}");

                if (Verbose)
                    LogLine($"Ep# {ep} reward(avg): {Math.Round(totalReward, 2)}({Math.Round(rewardAvg.Avg, 2)}) steps: {step} ε: {Math.Round(Epsilon, 4)} total_steps: {globalStep}");

                if (globalStep >= WarmupSteps && EpsilonDecayMode == EEpsilonDecayMode.EveryEpisode)
                    DecayEpsilon();

                if (ep % 10 == 0)
                    rewardChart.Save();
            }

            rewardChart.Save();
            SaveLog();
        }

        public float Test(int episodes, int maxStepsPerEpisode, int renderInterval)
        {
            var rewardAvg = new MovingAverage(episodes);

            for (int ep = 0; ep < episodes; ++ep)
            {
                bool render = renderInterval > 0 && (ep % renderInterval) == 0;

                LastObservation = Env.Reset();
                float totalReward = 0;
                int step = 0;
                for (; step < maxStepsPerEpisode; ++step)
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

                rewardAvg.Add(totalReward);

                if (Verbose)
                    LogLine($"Test# {ep} reward(avg): {Math.Round(totalReward, 2)}({Math.Round(rewardAvg.Avg, 2)}) steps: {step}");
            }

            SaveLog();
            return rewardAvg.Avg;
        }

        public virtual void Save(string filename) { }
        public virtual void Load(string filename) { }

        protected abstract Tensor GetOptimalAction();
        protected abstract void OnStep(int step, int globalStep, Tensor action, float reward, Tensor nextState, bool done);
        protected abstract void OnTrain();
        protected virtual void OnEpisodeEnd(int episode) { }

        protected virtual string GetParametersDescription()
        {
            return $"{GetType().Name}: ε_max={MaxEpsilon} ε_min={MinEpsilon} ε_decay/mode={EpsilonDecay}/{EpsilonDecayMode} train_int={TrainInterval} reward_clip={RewardClipping}";
        }

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
        public int TrainInterval = 1;        
        // Number of total steps that have to be performed before agent will start training
        public int WarmupSteps = 0;
        // Parameters training parameters will be saved every that number of episodes (0 for no saving)
        public int SaveFreq = 50;
        // Training episode will be rendered every that number of episodes (0 for no rendering)
        public int TrainRenderInterval = 0;
        // When not NaN, reward for step in which simulation ended will be overwritten with that value
        public float RewardOnDone = float.NaN;
        public bool RewardClipping = false;
        // Used for controlling rendering FPS
        public int RenderFreq = 30;        
        public bool Verbose = false;
        public int RewardAverageN = 100;
        public int StepsAverageN = 50;

        protected float Epsilon; // Exploration probability
        protected float RewardHighScore = float.MinValue;
        public readonly string Name;
        private List<string> LogLines = new List<string>();
    }
}
