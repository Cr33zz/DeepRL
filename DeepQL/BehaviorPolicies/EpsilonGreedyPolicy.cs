using DeepQL.Spaces;
using DeepQL.ValueFunc;
using Neuro.Tensors;

namespace DeepQL.BehaviorPolicies
{
    public class EpsilonGreedyPolicy : RandomPolicy
    {
        public EpsilonGreedyPolicy(Space actionSpate, ValueFunctionModel qFunc, float epsilon)
            : base(actionSpate)
        {
            QFunction = qFunc;
            Epsilon = epsilon;
        }

        public override Tensor GetNextAction(Tensor state)
        {
            if (Rand.NextDouble() < Epsilon)
                return base.GetNextAction(state);

            return QFunction.GetOptimalAction(state);
        }

        private ValueFunctionModel QFunction;
        public readonly float Epsilon;
    }
}
