using System;
using System.Collections.Generic;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class QTable : ValueFunctionModel
    {
        public QTable(int numberOfStates, int numberOfActions, double learningRate, double discountFactor)
            : base(new Shape(numberOfStates), numberOfActions, learningRate, discountFactor)
        {
            Table = new double[numberOfStates, numberOfActions];
        }

        public override Tensor GetOptimalAction(Tensor state)
        {
            var action = new Tensor(new Shape(1));
            action[0] = BestActionBasedOnQTable((int)state[0]);
            return action;
        }

        public override void OnTransition(Tensor state, Tensor action, double reward, Tensor nextState, bool done)
        {
            int stateInt = (int)state[0];
            int actionInt = (int)action[0];
            Table[stateInt, actionInt] += LearningRate * (reward + DiscountFactor * GetMaxRewardBasedOnQTable((int)nextState[0]) - Table[stateInt, actionInt]);
        }

        protected override void Train(List<Transition> transitions)
        {
            foreach (var trans in transitions)
                OnTransition(trans.State, trans.Action, trans.Reward, trans.NextState, trans.Done);
        }

        private int BestActionBasedOnQTable(int state)
        {
            double max = double.MinValue;
            int bestAction = -1;

            for (int a = 0; a < Table.GetLength(1); ++a)
            {
                if (Table[state, a] > max)
                {
                    max = Table[state, a];
                    bestAction = a;
                }
            }

            return bestAction;
        }

        private double GetMaxRewardBasedOnQTable(int state)
        {
            double max = double.MinValue;

            for (int a = 0; a < Table.GetLength(1); ++a)
                max = Math.Max(Table[state, a], max);

            return max;
        }
        
        private readonly double[,] Table;
    }
}
