using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepQL.Spaces;
using Neuro.Tensors;

namespace DeepQL.Environments
{
    public abstract class DiscreteEnv : Env
    {
        protected DiscreteEnv(int statesNum, int actionsNum)
            : base(new Discrete(actionsNum), new Discrete(statesNum))
        {
            InitialStateDistribution = new float[statesNum];
            TransitionsTable = new List<Transition>[statesNum, actionsNum];
            State = new Tensor(new Shape(1));
            LastAction = new Tensor(new Shape(1));
        }

        public override bool Step(Tensor action, out Tensor observation, out float reward)
        {
            var transitions = TransitionsTable[StateAsInt, (int)action[0]];
            int tInx = CategoricalSample(transitions.Select(x => x.Probability));

            var t = transitions[tInx];
            LastActionAsInt = (int)action[0];
            StateAsInt = t.NextState;
            observation = new Tensor(new float[] { t.NextState }, new Shape(1));
            reward = t.Reward;
            return t.Done;
        }

        public override Tensor Reset()
        {
            LastActionAsInt = -1;
            StateAsInt = CategoricalSample(InitialStateDistribution);
            return State.Clone();
        }

        protected void SetInitialStateDistribution(int state, float weight)
        {
            InitialStateDistribution[state] = weight;
        }

        protected void FinalizeInitialStateDistribution()
        {
            float sum = InitialStateDistribution.Sum();

            for (int i = 0; i < InitialStateDistribution.Length; ++i)
                InitialStateDistribution[i] /= sum;
        }

        protected void AddTransition(int state, int action, float probability, int nextState, float reward, bool done)
        {
            if (TransitionsTable[state, action] == null) TransitionsTable[state, action] = new List<Transition>();
            TransitionsTable[state, action].Add(new Transition() { Probability = 1.0f, NextState = nextState, Reward = reward, Done = done });
        }

        private int CategoricalSample(IEnumerable<float> probs)
        {
            // assuming all probabilities sum up to 1
            float p = (float)Rng.NextDouble();
            float probSum = 0;
            int probsNum = probs.Count();

            for (int i = 0; i < probsNum - 1; ++i)
            {
                probSum += probs.ElementAt(i);
                if (probSum >= p)
                    return i;
            }

            return probsNum - 1;
        }

        private struct Transition
        {
            public float Probability;
            public int NextState;
            public float Reward;
            public bool Done;
        }

        public int StateAsInt { get { return (int)State[0]; } private set { State[0] = value; } }
        public int LastActionAsInt { get { return (int)LastAction[0]; } private set { LastAction[0] = value; } }

        private readonly float[] InitialStateDistribution;
        private readonly List<Transition>[,] TransitionsTable;
    }
}
