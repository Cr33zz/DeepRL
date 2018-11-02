using Neuro.Tensors;
using System;

namespace DeepQL.BehaviorPolicies
{
    public class RandomPolicy : BasePolicy
    {
        public RandomPolicy(int numberOfActions) : base(numberOfActions) { }

        public override int GetNextAction(Tensor state)
        {
            return Rand.Next(0, NumberOfActions);
        }

        protected Random Rand = new Random();
    }
}
