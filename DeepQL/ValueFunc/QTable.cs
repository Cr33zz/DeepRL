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

        public override double GetOptimalAction(Tensor state)
        {
            return BestActionBasedOnQTable((int)state[0]);
        }

        public override void OnTransition(Tensor state, int action, double reward, Tensor nextState)
        {
            int stateInt = (int)state[0];
            Table[stateInt, action] += LearningRate * (reward + DiscountFactor * GetMaxRewardBasedOnQTable((int)nextState[0]) - Table[stateInt, action]);
        }

        protected override void Train(List<Transition> transitions)
        {
            foreach (var trans in transitions)
                OnTransition(trans.State, trans.Action, trans.Reward, trans.NextState);
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
