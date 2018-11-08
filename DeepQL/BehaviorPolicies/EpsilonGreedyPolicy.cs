using DeepQL.ValueFunc;
using Neuro.Tensors;
using System;

namespace DeepQL.BehaviorPolicies
{
    public class EpsilonGreedyPolicy : RandomPolicy
    {
        public EpsilonGreedyPolicy(int numberOfActions, ValueFunctionModel qFunc, double epsilon)
            : base(numberOfActions)
        {
            QFunction = qFunc;
            Epsilon = epsilon;
        }

        public override int GetNextAction(Tensor state)
        {
            if (Rand.NextDouble() < Epsilon)
                return base.GetNextAction(state);

            return (int)QFunction.GetOptimalAction(state);
        }

        private ValueFunctionModel QFunction;
        public readonly double Epsilon;
    }
}
