using DeepQL.Spaces;
using Neuro.Tensors;
using System;

namespace DeepQL.BehaviorPolicies
{
    public class RandomPolicy : BasePolicy
    {
        public RandomPolicy(Space actionSpate) : base(actionSpate) { }

        public override Tensor GetNextAction(Tensor state)
        {
            return ActionSpace.Sample();
        }

        protected Random Rand = new Random();
    }
}
