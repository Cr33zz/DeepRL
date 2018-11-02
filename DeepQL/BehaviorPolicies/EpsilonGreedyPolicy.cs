using DeepQL.Q;
using Neuro.Tensors;
using System;

namespace DeepQL.BehaviorPolicies
{
    public class EpsilonGreedyPolicy : RandomPolicy
    {
        public EpsilonGreedyPolicy(int numberOfActions, QFunc qFunc, double epsilon)
            : base(numberOfActions)
        {
            QFunction = qFunc;
            Epsilon = epsilon;
        }

        public override int GetNextAction(Tensor state)
        {
            if (Rand.NextDouble() < Epsilon)
                return base.GetNextAction(state);

            return QFunction.GetBestAction(state);
        }

        private QFunc QFunction;
        public readonly double Epsilon;
    }
}
